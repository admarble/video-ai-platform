from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
import aioredis
from pathlib import Path

class FailureReason(Enum):
    """Types of processing failures"""
    CORRUPTED_VIDEO = "corrupted_video"
    INVALID_FORMAT = "invalid_format"
    PROCESSING_ERROR = "processing_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    UNKNOWN = "unknown"

@dataclass
class FailedTask:
    """Represents a failed processing task"""
    task_id: str
    video_path: str
    failure_reason: FailureReason
    error_message: str
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class DeadLetterQueue:
    """Manages failed tasks and retry logic"""
    
    def __init__(
        self,
        redis_url: str,
        retry_intervals: Optional[List[int]] = None,
        max_retries: int = 3,
        failure_ttl: int = 604800  # 7 days
    ):
        self.redis_url = redis_url
        self.retry_intervals = retry_intervals or [300, 900, 3600]  # 5min, 15min, 1hour
        self.max_retries = max_retries
        self.failure_ttl = failure_ttl
        self.redis: Optional[aioredis.Redis] = None
        self.logger = logging.getLogger(__name__)
        
        # Keys for Redis
        self.failed_tasks_key = "video_processing:failed_tasks"
        self.retry_queue_key = "video_processing:retry_queue"
        
    async def init(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.from_url(self.redis_url)
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis:
            await self.redis.close()
            
    async def add_failed_task(self, task: FailedTask):
        """Add failed task to DLQ"""
        try:
            # Store task data
            task_data = {
                'task_id': task.task_id,
                'video_path': task.video_path,
                'failure_reason': task.failure_reason.value,
                'error_message': task.error_message,
                'timestamp': task.timestamp,
                'retry_count': task.retry_count,
                'max_retries': task.max_retries,
                'last_retry': task.last_retry,
                'metadata': task.metadata
            }
            
            # Add to failed tasks set
            await self.redis.hset(
                self.failed_tasks_key,
                task.task_id,
                json.dumps(task_data)
            )
            
            # Set expiry
            await self.redis.expire(self.failed_tasks_key, self.failure_ttl)
            
            # Add to retry queue if retries remain
            if task.retry_count < task.max_retries:
                retry_time = time.time() + self.retry_intervals[task.retry_count]
                await self.redis.zadd(self.retry_queue_key, {task.task_id: retry_time})
                
            self.logger.info(
                f"Added failed task {task.task_id} to DLQ "
                f"(retry {task.retry_count}/{task.max_retries})"
            )
            
        except Exception as e:
            self.logger.error(f"Error adding failed task: {str(e)}")
            
    async def get_failed_task(self, task_id: str) -> Optional[FailedTask]:
        """Get failed task details"""
        try:
            data = await self.redis.hget(self.failed_tasks_key, task_id)
            if not data:
                return None
                
            task_data = json.loads(data)
            return FailedTask(
                task_id=task_data['task_id'],
                video_path=task_data['video_path'],
                failure_reason=FailureReason(task_data['failure_reason']),
                error_message=task_data['error_message'],
                timestamp=task_data['timestamp'],
                retry_count=task_data['retry_count'],
                max_retries=task_data['max_retries'],
                last_retry=task_data['last_retry'],
                metadata=task_data['metadata']
            )
            
        except Exception as e:
            self.logger.error(f"Error getting failed task: {str(e)}")
            return None
            
    async def get_tasks_for_retry(self) -> List[FailedTask]:
        """Get tasks ready for retry"""
        try:
            current_time = time.time()
            
            # Get tasks with retry time <= current time
            task_ids = await self.redis.zrangebyscore(
                self.retry_queue_key,
                min=0,
                max=current_time
            )
            
            tasks = []
            for task_id in task_ids:
                task = await self.get_failed_task(task_id)
                if task:
                    tasks.append(task)
                    
            return tasks
            
        except Exception as e:
            self.logger.error(f"Error getting retry tasks: {str(e)}")
            return []
            
    async def retry_task(
        self,
        task: FailedTask,
        retry_handler: Callable[[FailedTask], Any]
    ) -> bool:
        """Retry failed task"""
        try:
            self.logger.info(f"Retrying task {task.task_id}")
            
            # Update retry count and timestamp
            task.retry_count += 1
            task.last_retry = time.time()
            
            # Remove from retry queue
            await self.redis.zrem(self.retry_queue_key, task.task_id)
            
            # Attempt retry
            try:
                await retry_handler(task)
                
                # Success - remove from failed tasks
                await self.redis.hdel(self.failed_tasks_key, task.task_id)
                self.logger.info(f"Successfully retried task {task.task_id}")
                return True
                
            except Exception as e:
                # Failed - update task and add back to DLQ if retries remain
                task.error_message = str(e)
                await self.add_failed_task(task)
                self.logger.warning(
                    f"Retry failed for task {task.task_id}: {str(e)}"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error in retry process: {str(e)}")
            return False
            
    async def get_failed_task_stats(self) -> Dict[str, Any]:
        """Get statistics about failed tasks"""
        try:
            # Get all failed tasks
            tasks_data = await self.redis.hgetall(self.failed_tasks_key)
            tasks = [
                json.loads(data) for data in tasks_data.values()
            ]
            
            # Calculate statistics
            total_tasks = len(tasks)
            if not total_tasks:
                return {
                    'total_tasks': 0,
                    'failure_reasons': {},
                    'retry_stats': {
                        'pending_retries': 0,
                        'max_retries_reached': 0,
                        'avg_retries': 0
                    }
                }
                
            # Count failure reasons
            failure_reasons = {}
            for task in tasks:
                reason = task['failure_reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                
            # Calculate retry statistics
            retry_counts = [task['retry_count'] for task in tasks]
            max_retries_reached = sum(
                1 for task in tasks
                if task['retry_count'] >= task['max_retries']
            )
            pending_retries = await self.redis.zcard(self.retry_queue_key)
            
            return {
                'total_tasks': total_tasks,
                'failure_reasons': failure_reasons,
                'retry_stats': {
                    'pending_retries': pending_retries,
                    'max_retries_reached': max_retries_reached,
                    'avg_retries': sum(retry_counts) / total_tasks
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting task stats: {str(e)}")
            return {}
            
    async def cleanup_old_tasks(self, max_age: int = 604800):
        """Remove old tasks from DLQ"""
        try:
            cutoff_time = time.time() - max_age
            
            # Get all tasks
            tasks_data = await self.redis.hgetall(self.failed_tasks_key)
            
            # Find old tasks
            old_task_ids = []
            for task_id, data in tasks_data.items():
                task_data = json.loads(data)
                if task_data['timestamp'] < cutoff_time:
                    old_task_ids.append(task_id)
                    
            if old_task_ids:
                # Remove from failed tasks and retry queue
                await self.redis.hdel(self.failed_tasks_key, *old_task_ids)
                await self.redis.zrem(self.retry_queue_key, *old_task_ids)
                
                self.logger.info(f"Cleaned up {len(old_task_ids)} old tasks")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old tasks: {str(e)}")

class DeadLetterQueueManager:
    """Manages DLQ operations and retry processing"""
    
    def __init__(
        self,
        dlq: DeadLetterQueue,
        retry_handler: Callable[[FailedTask], Any],
        check_interval: int = 60
    ):
        self.dlq = dlq
        self.retry_handler = retry_handler
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._retry_task = None
        
    async def start(self):
        """Start DLQ manager"""
        if self._running:
            return
            
        self._running = True
        self._retry_task = asyncio.create_task(self._process_retries())
        
    async def stop(self):
        """Stop DLQ manager"""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
                
    async def _process_retries(self):
        """Process retry queue"""
        while self._running:
            try:
                # Get tasks ready for retry
                tasks = await self.dlq.get_tasks_for_retry()
                
                for task in tasks:
                    # Attempt retry
                    await self.dlq.retry_task(task, self.retry_handler)
                    
            except Exception as e:
                self.logger.error(f"Error processing retries: {str(e)}")
                
            await asyncio.sleep(self.check_interval)

async def setup_dlq(redis_url: str) -> DeadLetterQueueManager:
    """Setup DLQ system"""
    # Create and initialize DLQ
    dlq = DeadLetterQueue(redis_url)
    await dlq.init()
    
    # Create manager with default retry handler
    async def default_retry_handler(task: FailedTask):
        raise NotImplementedError("Retry handler not implemented")
    
    manager = DeadLetterQueueManager(dlq, default_retry_handler)
    await manager.start()
    
    return manager 