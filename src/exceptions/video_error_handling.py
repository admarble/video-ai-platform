from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
import logging
import asyncio
from pathlib import Path
import traceback
import torch
import psutil

class VideoErrorType(Enum):
    """Detailed video processing error types"""
    # File-related errors
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    FILE_CORRUPTED = "file_corrupted"
    
    # Format errors
    INVALID_FORMAT = "invalid_format"
    UNSUPPORTED_CODEC = "unsupported_codec"
    INVALID_RESOLUTION = "invalid_resolution"
    MISSING_VIDEO_STREAM = "missing_video_stream"
    MISSING_AUDIO_STREAM = "missing_audio_stream"
    
    # Processing errors
    FRAME_EXTRACTION_ERROR = "frame_extraction_error"
    SCENE_DETECTION_ERROR = "scene_detection_error"
    OBJECT_DETECTION_ERROR = "object_detection_error"
    AUDIO_PROCESSING_ERROR = "audio_processing_error"
    
    # Resource errors
    INSUFFICIENT_MEMORY = "insufficient_memory"
    GPU_ERROR = "gpu_error"
    DISK_SPACE_ERROR = "disk_space_error"
    
    # Model errors
    MODEL_LOAD_ERROR = "model_load_error"
    MODEL_INFERENCE_ERROR = "model_inference_error"
    MODEL_TIMEOUT = "model_timeout"
    
    # System errors
    PROCESS_KILLED = "process_killed"
    SYSTEM_OVERLOAD = "system_overload"
    NETWORK_ERROR = "network_error"
    
    # Generic errors
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class VideoProcessingError:
    """Detailed video processing error information"""
    error_type: VideoErrorType
    message: str
    video_path: str
    timestamp: float
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    recovery_hint: Optional[str] = None

class ErrorCategory(Enum):
    """Categories of errors for retry strategy"""
    TRANSIENT = "transient"  # Temporary issues that may resolve themselves
    RECOVERABLE = "recoverable"  # Issues that might be fixed by retrying
    PERMANENT = "permanent"  # Fatal errors that won't be fixed by retrying

@dataclass
class RetryStrategy:
    """Strategy for retrying failed tasks"""
    max_retries: int
    base_delay: int  # seconds
    max_delay: int  # seconds
    backoff_factor: float = 2.0
    jitter: bool = True
    requires_intervention: bool = False
    cleanup_required: bool = False

@dataclass
class FailedTask:
    """Represents a failed video processing task"""
    task_id: str
    video_path: str
    failure_reason: str
    error_message: str
    timestamp: float
    max_retries: int
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class VideoErrorHandler:
    """Handles video processing errors with specific strategies"""
    
    def __init__(self, dlq_manager: 'DeadLetterQueueManager'):
        self.dlq_manager = dlq_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize error categorization and retry strategies
        self._init_error_strategies()
        
    def _init_error_strategies(self):
        """Initialize error handling strategies"""
        # Define retry strategies for different categories
        self.retry_strategies = {
            ErrorCategory.TRANSIENT: RetryStrategy(
                max_retries=5,
                base_delay=60,   # 1 minute
                max_delay=3600,  # 1 hour
                backoff_factor=2.0,
                jitter=True
            ),
            ErrorCategory.RECOVERABLE: RetryStrategy(
                max_retries=3,
                base_delay=300,  # 5 minutes
                max_delay=7200,  # 2 hours
                backoff_factor=2.0,
                jitter=True,
                cleanup_required=True
            ),
            ErrorCategory.PERMANENT: RetryStrategy(
                max_retries=0,
                base_delay=0,
                max_delay=0,
                requires_intervention=True
            )
        }
        
        # Categorize error types
        self.error_categories = {
            # Transient errors
            VideoErrorType.INSUFFICIENT_MEMORY: ErrorCategory.TRANSIENT,
            VideoErrorType.GPU_ERROR: ErrorCategory.TRANSIENT,
            VideoErrorType.SYSTEM_OVERLOAD: ErrorCategory.TRANSIENT,
            VideoErrorType.NETWORK_ERROR: ErrorCategory.TRANSIENT,
            VideoErrorType.TIMEOUT: ErrorCategory.TRANSIENT,
            VideoErrorType.PROCESS_KILLED: ErrorCategory.TRANSIENT,
            
            # Recoverable errors
            VideoErrorType.FRAME_EXTRACTION_ERROR: ErrorCategory.RECOVERABLE,
            VideoErrorType.SCENE_DETECTION_ERROR: ErrorCategory.RECOVERABLE,
            VideoErrorType.OBJECT_DETECTION_ERROR: ErrorCategory.RECOVERABLE,
            VideoErrorType.AUDIO_PROCESSING_ERROR: ErrorCategory.RECOVERABLE,
            VideoErrorType.MODEL_LOAD_ERROR: ErrorCategory.RECOVERABLE,
            VideoErrorType.MODEL_TIMEOUT: ErrorCategory.RECOVERABLE,
            
            # Permanent errors
            VideoErrorType.FILE_NOT_FOUND: ErrorCategory.PERMANENT,
            VideoErrorType.PERMISSION_ERROR: ErrorCategory.PERMANENT,
            VideoErrorType.FILE_CORRUPTED: ErrorCategory.PERMANENT,
            VideoErrorType.INVALID_FORMAT: ErrorCategory.PERMANENT,
            VideoErrorType.UNSUPPORTED_CODEC: ErrorCategory.PERMANENT,
            VideoErrorType.INVALID_RESOLUTION: ErrorCategory.PERMANENT,
            VideoErrorType.MISSING_VIDEO_STREAM: ErrorCategory.PERMANENT,
            VideoErrorType.MISSING_AUDIO_STREAM: ErrorCategory.PERMANENT
        }
        
    def _get_recovery_hint(self, error_type: VideoErrorType) -> str:
        """Get recovery hint for error type"""
        hints = {
            VideoErrorType.INSUFFICIENT_MEMORY: "Try reducing batch size or freeing system memory",
            VideoErrorType.GPU_ERROR: "Check GPU health and memory usage",
            VideoErrorType.DISK_SPACE_ERROR: "Free up disk space",
            VideoErrorType.FILE_CORRUPTED: "Video file may be corrupted, try re-uploading",
            VideoErrorType.INVALID_FORMAT: "Video format not supported, convert to supported format",
            VideoErrorType.UNSUPPORTED_CODEC: "Convert video to supported codec",
            VideoErrorType.MODEL_LOAD_ERROR: "Check model files and GPU memory",
            VideoErrorType.NETWORK_ERROR: "Check network connectivity",
            VideoErrorType.TIMEOUT: "Try increasing timeout duration"
        }
        return hints.get(error_type, "Contact system administrator for assistance")
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        info = {
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count()
            },
            'disk': {
                'usage': psutil.disk_usage('/').percent
            }
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            info['gpu'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0)
            }
            
        return info
        
    async def handle_error(
        self,
        error: Exception,
        video_path: str,
        task_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> VideoProcessingError:
        """Handle video processing error"""
        try:
            # Determine error type
            error_type = self._classify_error(error)
            
            # Create error info
            error_info = VideoProcessingError(
                error_type=error_type,
                message=str(error),
                video_path=video_path,
                timestamp=time.time(),
                error_details=context,
                stack_trace=traceback.format_exc(),
                system_info=self._get_system_info(),
                recovery_hint=self._get_recovery_hint(error_type)
            )
            
            # Get error category and strategy
            category = self.error_categories.get(error_type, ErrorCategory.PERMANENT)
            strategy = self.retry_strategies[category]
            
            # Create failed task
            failed_task = FailedTask(
                task_id=task_id,
                video_path=video_path,
                failure_reason=error_type.value,
                error_message=str(error),
                timestamp=time.time(),
                max_retries=strategy.max_retries,
                metadata={
                    'error_info': asdict(error_info),
                    'category': category.value,
                    'requires_intervention': strategy.requires_intervention,
                    'cleanup_required': strategy.cleanup_required
                }
            )
            
            # Add to DLQ with appropriate retry strategy
            await self.dlq_manager.dlq.add_failed_task(
                failed_task,
                retry_delay=self._calculate_retry_delay(
                    strategy,
                    failed_task.retry_count
                )
            )
            
            # Log error
            self._log_error(error_info, category)
            
            return error_info
            
        except Exception as e:
            self.logger.error(f"Error in error handler: {str(e)}")
            raise
            
    def _classify_error(self, error: Exception) -> VideoErrorType:
        """Classify exception into VideoErrorType"""
        if isinstance(error, FileNotFoundError):
            return VideoErrorType.FILE_NOT_FOUND
        elif isinstance(error, PermissionError):
            return VideoErrorType.PERMISSION_ERROR
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            return VideoErrorType.GPU_ERROR
        elif isinstance(error, MemoryError):
            return VideoErrorType.INSUFFICIENT_MEMORY
        elif isinstance(error, TimeoutError):
            return VideoErrorType.TIMEOUT
        
        # Check error message for specific patterns
        error_msg = str(error).lower()
        
        if "corrupt" in error_msg:
            return VideoErrorType.FILE_CORRUPTED
        elif "codec" in error_msg:
            return VideoErrorType.UNSUPPORTED_CODEC
        elif "resolution" in error_msg:
            return VideoErrorType.INVALID_RESOLUTION
        elif "network" in error_msg:
            return VideoErrorType.NETWORK_ERROR
        elif "model" in error_msg:
            return VideoErrorType.MODEL_INFERENCE_ERROR
        
        return VideoErrorType.UNKNOWN
        
    def _calculate_retry_delay(
        self,
        strategy: RetryStrategy,
        retry_count: int
    ) -> int:
        """Calculate delay before next retry"""
        if retry_count >= strategy.max_retries:
            return -1  # No more retries
            
        # Calculate exponential backoff
        delay = min(
            strategy.base_delay * (strategy.backoff_factor ** retry_count),
            strategy.max_delay
        )
        
        # Add jitter if enabled
        if strategy.jitter:
            import random
            delay *= (0.5 + random.random())
            
        return int(delay)
        
    def _log_error(
        self,
        error_info: VideoProcessingError,
        category: ErrorCategory
    ):
        """Log error with appropriate level"""
        if category == ErrorCategory.PERMANENT:
            level = logging.ERROR
        elif category == ErrorCategory.RECOVERABLE:
            level = logging.WARNING
        else:
            level = logging.INFO
            
        self.logger.log(
            level,
            f"Video processing error: {error_info.error_type.value}",
            extra={
                'video_path': error_info.video_path,
                'error_type': error_info.error_type.value,
                'category': category.value,
                'system_info': error_info.system_info
            }
        )

class VideoProcessingRetryHandler:
    """Handles retrying failed video processing tasks"""
    
    def __init__(
        self,
        processor: 'VideoProcessor',
        error_handler: VideoErrorHandler
    ):
        self.processor = processor
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
    async def handle_retry(self, task: FailedTask) -> bool:
        """Handle retry of failed task"""
        try:
            # Check if cleanup is needed
            if task.metadata.get('cleanup_required'):
                await self._cleanup_failed_task(task)
                
            # Check if manual intervention is required
            if task.metadata.get('requires_intervention'):
                self.logger.warning(
                    f"Task {task.task_id} requires manual intervention"
                )
                return False
                
            # Attempt to process video again
            await self.processor.process_video(
                task.video_path,
                task_id=task.task_id
            )
            
            self.logger.info(f"Successfully retried task {task.task_id}")
            return True
            
        except Exception as e:
            # Handle retry failure
            await self.error_handler.handle_error(
                e,
                task.video_path,
                task.task_id,
                context={'retry_attempt': task.retry_count + 1}
            )
            return False
            
    async def _cleanup_failed_task(self, task: FailedTask):
        """Clean up resources from failed task"""
        try:
            # Clean up temporary files
            temp_dir = Path(f"temp/{task.task_id}")
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                
            # Clean up any GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up task {task.task_id}: {str(e)}")

async def setup_video_error_handling(
    dlq_manager: 'DeadLetterQueueManager',
    processor: 'VideoProcessor'
) -> Tuple[VideoErrorHandler, VideoProcessingRetryHandler]:
    """Setup video error handling system"""
    error_handler = VideoErrorHandler(dlq_manager)
    retry_handler = VideoProcessingRetryHandler(processor, error_handler)
    
    return error_handler, retry_handler 