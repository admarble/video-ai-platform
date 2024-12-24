API Reference
============

This section provides detailed API documentation for the Cuthrough library.

Core Components
-------------

.. module:: src.processors.adaptive_compression

Adaptive Compression
~~~~~~~~~~~~~~~~~~

.. autoclass:: CompressionProfile
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CompressionMetric
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OptimizationStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OptimizationConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: create_compression_tuner

FFmpeg Integration
---------------

.. module:: src.processors.ffmpeg_processor

.. autoclass:: FFmpegProcessor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: VideoMetadata
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
~~~~~~~~~

.. autoexception:: FFmpegError
   :members:
   :show-inheritance:

.. autoexception:: FFmpegNotFoundError
   :members:
   :show-inheritance:

Utility Functions
--------------

.. module:: src.utils.logging_config

Logging Configuration
~~~~~~~~~~~~~~~~~~

.. autofunction:: setup_logging

.. autofunction:: get_logger

Command Line Interface
-------------------

.. module:: src.cuthrough

Main CLI
~~~~~~~

.. autofunction:: main

.. autofunction:: create_parser

Command Handlers
~~~~~~~~~~~~~

.. autofunction:: compress_command

.. autofunction:: analyze_command

.. autofunction:: profile_command

Helper Functions
~~~~~~~~~~~~~

.. autofunction:: setup_common_args

.. autofunction:: validate_args

.. autofunction:: print_progress

.. autofunction:: create_optimization_config

.. autofunction:: create_compression_profile

Type Hints
--------

The library uses type hints extensively. Here are the main types used:

.. code-block:: python

   from typing import Dict, Any, Optional, List, Tuple, Union

   # Common type aliases
   PathLike = Union[str, Path]
   JsonDict = Dict[str, Any]
   MetricsDict = Dict[CompressionMetric, float]
   ProgressCallback = Optional[Callable[[float], None]]

Examples
-------

Creating a Compression Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.processors.adaptive_compression import (
       CompressionProfile,
       VideoCodec,
       AudioCodec,
       ContainerFormat
   )

   profile = CompressionProfile(
       name="high_quality",
       video_codec=VideoCodec.H264,
       audio_codec=AudioCodec.AAC,
       container_format=ContainerFormat.MP4,
       video_bitrate="5M",
       audio_bitrate="192k",
       preset="veryslow",
       quality_value=18,
       multipass=True
   )

Using the FFmpeg Processor
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.processors.ffmpeg_processor import FFmpegProcessor
   from pathlib import Path

   # Initialize processor
   ffmpeg = FFmpegProcessor()

   # Get video information
   video_info = ffmpeg.get_video_info(Path("input.mp4"))

   # Compress video
   metrics = ffmpeg.compress_video(
       Path("input.mp4"),
       Path("output.mp4"),
       profile,
       progress_callback=lambda t: print(f"Progress: {t:.1f}s")
   )

Setting Up Logging
~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils.logging_config import setup_logging, get_logger
   from pathlib import Path

   # Set up logging
   setup_logging(
       log_level="DEBUG",
       log_file=Path("compression.log")
   )

   # Get logger
   logger = get_logger(__name__)
   logger.info("Starting compression") 