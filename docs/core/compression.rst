Compression Middleware
====================

The compression middleware provides efficient HTTP compression with multiple algorithms, smart decision-making, and streaming support.

Features
--------

- Multiple compression methods (Gzip, Deflate, Brotli)
- Smart compression decisions based on:
    - Minimum size thresholds
    - Content type exclusions
    - Path exclusions
    - Client capability detection
- Streaming support with buffer-based compression
- Comprehensive metrics tracking
- Configurable preferences and fallbacks

Installation
-----------

The compression middleware requires the ``brotli`` package. It's included in the project's requirements.txt:

.. code-block:: bash

    pip install -r requirements.txt

Basic Usage
----------

Create and use the middleware with default configuration:

.. code-block:: python

    from src.compression import create_compression_middleware

    # Create with default configuration
    middleware = create_compression_middleware()

    # Compress response data
    compressed_data, new_headers = middleware.compress_response(
        response_data,
        original_headers
    )

Custom Configuration
------------------

Customize the middleware behavior with configuration options:

.. code-block:: python

    config = {
        'enabled': True,
        'min_size': 2048,  # 2KB minimum
        'level': 7,        # Higher compression
        'excluded_paths': ['/api/stream', '/api/download'],
        'excluded_types': ['image/', 'video/']
    }
    middleware = create_compression_middleware(config)

Streaming Support
---------------

For large responses, use streaming compression:

.. code-block:: python

    from src.compression import StreamingCompressor, CompressionMethod

    # Create streaming compressor
    streamer = StreamingCompressor(
        middleware,
        CompressionMethod.GZIP
    )

    # Process stream chunks
    for chunk in data_stream:
        compressed_chunk = streamer.compress_chunk(chunk)
        if compressed_chunk:
            yield compressed_chunk

    # Flush remaining data
    final_chunk = streamer.flush()
    if final_chunk:
        yield final_chunk

Configuration Options
-------------------

CompressionConfig Parameters:
    - ``enabled`` (bool): Enable/disable compression
    - ``preferred_methods`` (List[CompressionMethod]): Ordered list of preferred compression methods
    - ``min_size`` (int): Minimum size in bytes to compress
    - ``level`` (int): Compression level (1-9)
    - ``excluded_paths`` (List[str]): URL paths to exclude from compression
    - ``excluded_types`` (List[str]): Content types to exclude from compression
    - ``buffer_size`` (int): Buffer size for streaming compression

Compression Methods
-----------------

Available compression methods:
    - ``CompressionMethod.GZIP``: Standard gzip compression
    - ``CompressionMethod.DEFLATE``: Deflate compression
    - ``CompressionMethod.BROTLI``: Brotli compression (typically best compression ratio)
    - ``CompressionMethod.NONE``: No compression

Metrics and Monitoring
--------------------

The middleware tracks comprehensive compression metrics:

.. code-block:: python

    stats = middleware.get_compression_stats()

Available metrics:
    - Total bytes before/after compression
    - Number of requests compressed/skipped
    - Usage statistics for each compression method
    - Overall compression ratio
    - Bytes saved
    - Compression rate

Headers Handling
--------------

The middleware automatically:
    - Checks Accept-Encoding header for client support
    - Sets appropriate Content-Encoding header
    - Updates Content-Length header
    - Removes Accept-Ranges header for compressed content

Future Enhancements
-----------------

Planned improvements:
    - Auto-tuning compression levels based on CPU usage
    - Memory limits for streaming compression
    - Async support for async web frameworks
    - Cache headers optimization
    - Vary header handling

Error Handling
------------

The middleware includes robust error handling:
    - Falls back to uncompressed data on compression errors
    - Logs compression errors with detailed information
    - Maintains metrics for skipped compressions
    - Safely handles invalid input data 