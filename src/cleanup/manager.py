from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
import time
import shutil
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import os

from .priority import CleanupPriority
from .task import CleanupTask

class CleanupManager:
    """Manages system cleanup routines"""
    
    def __init__(
        self,
        base_dir: Path,
        cache_manager: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.base_dir = base_dir
        self.cache_manager = cache_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize task registry
        self.tasks: Dict[str, CleanupTask] = {}
        
        # Setup locks and executor
        self.task_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_cleanup_workers', 4)
        )
        
        # For graceful shutdown
        self._shutdown = False
        self._scheduler_task = None
        
        # Initialize default tasks
        self._setup_default_tasks()
        
        # Start cleanup scheduler
        self._start_scheduler()
        
    def _setup_default_tasks(self):
        """Setup default cleanup tasks"""
        self.add_task(
            name="temp_files",
            priority=CleanupPriority.HIGH,
            interval=3600,  # Every hour
            func=self._cleanup_temp_files
        )
        
        self.add_task(
            name="processed_videos",
            priority=CleanupPriority.MEDIUM,
            interval=86400,  # Daily
            func=self._cleanup_processed_videos
        )
        
        self.add_task(
            name="failed_jobs",
            priority=CleanupPriority.LOW,
            interval=43200,  # Twice daily
            func=self._cleanup_failed_jobs
        )
        
        self.add_task(
            name="cache",
            priority=CleanupPriority.MEDIUM,
            interval=7200,  # Every 2 hours
            func=self._cleanup_cache
        )
        
        self.add_task(
            name="logs",
            priority=CleanupPriority.LOW,
            interval=86400,  # Daily
            func=self._cleanup_logs
        )
        
    def add_task(
        self,
        name: str,
        priority: CleanupPriority,
        interval: int,
        func: callable
    ):
        """Add new cleanup task"""
        with self.task_lock:
            self.tasks[name] = CleanupTask(
                name=name,
                priority=priority,
                interval=interval,
                func=func,
                last_run=0,
                enabled=True
            )
            
    def remove_task(self, name: str):
        """Remove cleanup task"""
        with self.task_lock:
            if name in self.tasks:
                del self.tasks[name]
                
    async def _cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = self.base_dir / "temp"
        if not temp_dir.exists():
            return
            
        current_time = time.time()
        max_age = self.config.get('temp_file_max_age', 3600)  # 1 hour
        
        files_to_clean = []
        total_files = 0
        
        for file_path in temp_dir.glob("*"):
            total_files += 1
            if current_time - file_path.stat().st_mtime > max_age:
                files_to_clean.append(file_path)

        task = self.tasks["temp_files"]
        for i, file_path in enumerate(files_to_clean):
            if self._shutdown:
                break
                
            try:
                task.progress = (i / len(files_to_clean)) * 100 if files_to_clean else 100
                await self._cleanup_with_metrics(
                    task,
                    lambda: self._delete_path(file_path),
                    file_path
                )
            except Exception as e:
                self.logger.error(f"Error cleaning temp file {file_path}: {str(e)}")

        task.progress = 100

    async def _delete_path(self, path: Path):
        """Delete file or directory"""
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    async def _cleanup_processed_videos(self):
        """Clean up processed video files"""
        processed_dir = self.base_dir / "processed"
        if not processed_dir.exists():
            return
            
        retention_days = self.config.get('processed_video_retention_days', 30)
        max_age = retention_days * 86400
        current_time = time.time()
        
        dirs_to_clean = []
        for video_dir in processed_dir.glob("*"):
            if not video_dir.is_dir():
                continue
                
            # Check if processing is complete
            status_file = video_dir / "status.json"
            if not status_file.exists():
                continue
                
            # Check age
            if current_time - video_dir.stat().st_mtime > max_age:
                dirs_to_clean.append(video_dir)

        task = self.tasks["processed_videos"]
        for i, video_dir in enumerate(dirs_to_clean):
            if self._shutdown:
                break
                
            try:
                task.progress = (i / len(dirs_to_clean)) * 100 if dirs_to_clean else 100
                await self._cleanup_with_metrics(
                    task,
                    lambda: self._delete_path(video_dir),
                    video_dir
                )
            except Exception as e:
                self.logger.error(f"Error cleaning processed video {video_dir}: {str(e)}")

        task.progress = 100
                
    async def _cleanup_failed_jobs(self):
        """Clean up failed job records"""
        failed_dir = self.base_dir / "failed"
        if not failed_dir.exists():
            return
            
        max_age = self.config.get('failed_job_retention_days', 7) * 86400
        current_time = time.time()
        
        files_to_clean = []
        for job_file in failed_dir.glob("*.json"):
            if current_time - job_file.stat().st_mtime > max_age:
                files_to_clean.append(job_file)

        task = self.tasks["failed_jobs"]
        for i, job_file in enumerate(files_to_clean):
            if self._shutdown:
                break
                
            try:
                task.progress = (i / len(files_to_clean)) * 100 if files_to_clean else 100
                await self._cleanup_with_metrics(
                    task,
                    lambda: job_file.unlink(),
                    job_file
                )
            except Exception as e:
                self.logger.error(f"Error cleaning failed job {job_file}: {str(e)}")

        task.progress = 100
                
    async def _cleanup_cache(self):
        """Clean up cached data"""
        if not self.cache_manager:
            return

        task = self.tasks["cache"]
        task.progress = 0
            
        try:
            task._is_running = True
            start_time = time.time()
            
            # Get cache size before cleanup
            cache_size = await self.cache_manager.get_cache_size()
            
            # Perform cleanup
            await self.cache_manager.cleanup_old_metrics()
            
            # Get cache size after cleanup
            new_cache_size = await self.cache_manager.get_cache_size()
            bytes_cleaned = max(0, cache_size - new_cache_size)
            
            execution_time = time.time() - start_time
            task.update_metrics(execution_time, bytes_cleaned)
            
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {str(e)}")
            task.update_metrics(time.time() - start_time, 0, failed=True)
        finally:
            task._is_running = False
            task.progress = 100
                
    async def _cleanup_logs(self):
        """Clean up old log files"""
        log_dir = self.base_dir / "logs"
        if not log_dir.exists():
            return
            
        max_age = self.config.get('log_retention_days', 30) * 86400
        current_time = time.time()
        
        files_to_clean = []
        for log_file in log_dir.glob("*.log*"):
            if current_time - log_file.stat().st_mtime > max_age:
                files_to_clean.append(log_file)

        task = self.tasks["logs"]
        for i, log_file in enumerate(files_to_clean):
            if self._shutdown:
                break
                
            try:
                task.progress = (i / len(files_to_clean)) * 100 if files_to_clean else 100
                await self._cleanup_with_metrics(
                    task,
                    lambda: log_file.unlink(),
                    log_file
                )
            except Exception as e:
                self.logger.error(f"Error cleaning log file {log_file}: {str(e)}")

        task.progress = 100

    def _start_scheduler(self):
        """Start cleanup scheduler"""
        async def scheduler():
            while not self._shutdown:
                try:
                    current_time = time.time()
                    
                    # Sort tasks by priority
                    sorted_tasks = sorted(
                        self.tasks.values(),
                        key=lambda x: (x.priority, -x.last_run),
                        reverse=True
                    )
                    
                    for task in sorted_tasks:
                        if (not task.enabled or 
                            current_time - task.last_run < task.interval or
                            task._is_running):  # Skip if already running
                            continue
                            
                        # Run cleanup task
                        try:
                            await task.func()
                            task.last_run = current_time
                            task.retry_count = 0
                        except Exception as e:
                            self.logger.error(
                                f"Error in cleanup task {task.name}: {str(e)}"
                            )
                            task.retry_count += 1
                            
                            if task.retry_count >= task.max_retries:
                                task.enabled = False
                                self.logger.error(
                                    f"Disabled cleanup task {task.name} after "
                                    f"{task.max_retries} failed attempts"
                                )
                                    
                except Exception as e:
                    self.logger.error(f"Error in cleanup scheduler: {str(e)}")
                    
                await asyncio.sleep(60)  # Check every minute
                
        # Start scheduler in background
        self._scheduler_task = asyncio.create_task(scheduler())
        
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of cleanup tasks"""
        with self.task_lock:
            return {
                name: {
                    'priority': task.priority.value,
                    'interval': task.interval,
                    'last_run': task.last_run,
                    'enabled': task.enabled,
                    'retry_count': task.retry_count,
                    **task.get_metrics()
                }
                for name, task in self.tasks.items()
            }
            
    async def cleanup_all(self, priority: Optional[CleanupPriority] = None):
        """Run all cleanup tasks immediately"""
        with self.task_lock:
            tasks = sorted(
                self.tasks.values(),
                key=lambda x: x.priority,
                reverse=True
            )
            
            for task in tasks:
                if priority and task.priority != priority:
                    continue
                if not task.enabled:
                    continue
                    
                try:
                    await task.func()
                    task.last_run = time.time()
                except Exception as e:
                    self.logger.error(
                        f"Error in cleanup task {task.name}: {str(e)}"
                    )

    async def shutdown(self):
        """Gracefully shutdown the cleanup manager"""
        self._shutdown = True
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Wait for running tasks to complete
        with self.task_lock:
            for task in self.tasks.values():
                if task._is_running:
                    await asyncio.sleep(0.1)  # Give tasks time to finish
        
        self.executor.shutdown(wait=True)

    def _get_file_size(self, path: Path) -> int:
        """Get size of file or directory in bytes"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = Path(dirpath) / f
                    total += fp.stat().st_size
            return total
        return 0

    async def _cleanup_with_metrics(self, task: CleanupTask, func, *paths: Path):
        """Run cleanup with metrics tracking"""
        start_time = time.time()
        bytes_cleaned = 0
        failed = False

        try:
            # Calculate size before cleanup
            for path in paths:
                if path.exists():
                    bytes_cleaned += self._get_file_size(path)

            task._is_running = True
            await func()
            
        except Exception as e:
            failed = True
            raise e
        finally:
            task._is_running = False
            execution_time = time.time() - start_time
            task.update_metrics(execution_time, bytes_cleaned, failed)

def create_cleanup_manager(
    base_dir: Path,
    cache_manager: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> CleanupManager:
    """Create cleanup manager instance"""
    return CleanupManager(
        base_dir=base_dir,
        cache_manager=cache_manager,
        config=config
    ) 