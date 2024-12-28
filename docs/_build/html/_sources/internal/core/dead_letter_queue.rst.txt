Dead Letter Queue System
====================

.. warning::
   This documentation is for internal use only. It contains implementation details that should not be exposed publicly.

The Dead Letter Queue (DLQ) system provides robust error handling and retry mechanisms for failed video processing tasks. It uses Redis as a backend store and supports configurable retry policies, task tracking, and comprehensive monitoring.

Key Features
-----------

- Redis-backed persistent storage
- Configurable retry intervals and policies
- Automatic task cleanup
- Comprehensive failure tracking
- Statistical monitoring
- Async/await support

Components
----------

FailureReason
~~~~~~~~~~~~

An enumeration of possible failure types:

.. code-block:: python

    class FailureReason(Enum):
        CORRUPTED_VIDEO = "corrupted_video"
        INVALID_FORMAT = "invalid_format"
        PROCESSING_ERROR = "processing_error"
        RESOURCE_EXHAUSTED = "resource_exhausted"
        TIMEOUT = "timeout"
        MODEL_ERROR = "model_error"
        UNKNOWN = "unknown"

FailedTask
~~~~~~~~~

A dataclass representing a failed processing task:

.. code-block:: python

    @dataclass
    class FailedTask:
        task_id: str
        video_path: str
        failure_reason: FailureReason
        error_message: str
        timestamp: float
        retry_count: int = 0
        max_retries: int = 3
        last_retry: Optional[float] = None
        metadata: Optional[Dict[str, Any]] = None

Implementation Details
--------------------

Redis Schema
~~~~~~~~~~~

The DLQ uses the following Redis keys and data structures:

- ``video_processing:failed_tasks`` (Hash): Stores failed task details
- ``video_processing:retry_queue`` (Sorted Set): Manages retry scheduling

Configuration
~~~~~~~~~~~~

The DLQ system supports several configuration options:

.. code-block:: python

    DeadLetterQueue(
        redis_url: str,
        retry_intervals: List[int] = [300, 900, 3600],  # 5min, 15min, 1hour
        max_retries: int = 3,
        failure_ttl: int = 604800  # 7 days
    )

- ``retry_intervals``: List of intervals (in seconds) between retry attempts
- ``max_retries``: Maximum number of retry attempts per task
- ``failure_ttl``: Time-to-live for failed tasks in seconds

Monitoring and Statistics
-----------------------

The DLQ system provides comprehensive monitoring capabilities:

.. code-block:: python

    # Get DLQ statistics
    stats = await dlq.get_failed_task_stats()

    # Example stats output:
    {
        'total_tasks': 100,
        'failure_reasons': {
            'corrupted_video': 20,
            'processing_error': 50,
            'timeout': 30
        },
        'retry_stats': {
            'pending_retries': 45,
            'max_retries_reached': 10,
            'avg_retries': 1.5
        }
    }

Cleanup and Maintenance
---------------------

The system includes automatic cleanup of old tasks:

.. code-block:: python

    # Clean up tasks older than 7 days
    await dlq.cleanup_old_tasks(max_age=604800)

Best Practices
-------------

1. **Error Classification**
   - Properly categorize errors using appropriate ``FailureReason`` values
   - Include detailed error messages for debugging

2. **Retry Intervals**
   - Use exponential backoff for retry intervals
   - Consider resource availability when setting intervals

3. **Monitoring**
   - Regularly check DLQ statistics
   - Set up alerts for tasks reaching max retries
   - Monitor retry success rates

4. **Resource Management**
   - Implement cleanup routines for old tasks
   - Set appropriate TTL values based on business needs

5. **Retry Handlers**
   - Implement idempotent retry handlers
   - Include proper error handling in retry logic
   - Log retry attempts and outcomes

Error Handling
-------------

The DLQ system provides comprehensive error handling:

.. code-block:: python

    try:
        # Your processing code
        process_video(video_path)
    except Exception as e:
        # Create and add failed task
        failed_task = FailedTask(
            task_id=task_id,
            video_path=video_path,
            failure_reason=FailureReason.PROCESSING_ERROR,
            error_message=str(e),
            timestamp=time.time()
        )
        await dlq.add_failed_task(failed_task)

API Reference
------------

DeadLetterQueue
~~~~~~~~~~~~~~

.. py:class:: DeadLetterQueue

    .. py:method:: add_failed_task(task: FailedTask)
        
        Add a failed task to the DLQ.

    .. py:method:: get_failed_task(task_id: str) -> Optional[FailedTask]
        
        Retrieve a specific failed task by ID.

    .. py:method:: get_tasks_for_retry() -> List[FailedTask]
        
        Get all tasks that are ready for retry.

    .. py:method:: retry_task(task: FailedTask, retry_handler: Callable) -> bool
        
        Attempt to retry a failed task.

    .. py:method:: get_failed_task_stats() -> Dict[str, Any]
        
        Get statistics about failed tasks.

    .. py:method:: cleanup_old_tasks(max_age: int = 604800)
        
        Remove tasks older than max_age seconds.

DeadLetterQueueManager
~~~~~~~~~~~~~~~~~~~~

.. py:class:: DeadLetterQueueManager

    .. py:method:: start()
        
        Start the DLQ manager and retry processing.

    .. py:method:: stop()
        
        Stop the DLQ manager and retry processing. 