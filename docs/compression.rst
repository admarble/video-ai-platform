Compression Features
===================

.. _adaptive-optimization:

Adaptive Optimization System
--------------------------

Cuthrough uses an advanced adaptive optimization system that automatically tunes compression parameters to achieve the best balance between quality, speed, and file size.

.. _optimization-strategies:

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~

.. py:class:: OptimizationStrategy

   .. py:attribute:: SIMPLE

      Basic parameter adjustment based on target metrics.
      
      - Good for quick compressions
      - Minimal computational overhead
      
      .. code-block:: bash
         
         cuthrough compress input.mp4 output.mp4 \
             --optimization-strategy SIMPLE \
             --quality 0.8

   .. py:attribute:: GRADIENT

      Gradient-based parameter optimization.
      
      - Better for finding optimal settings
      - Uses historical performance data
      
      .. code-block:: bash
         
         cuthrough compress input.mp4 output.mp4 \
             --optimization-strategy GRADIENT \
             --learning-rate 0.1 \
             --history-weight 0.9

   .. py:attribute:: ADAPTIVE

      Dynamic parameter tuning based on video content.
      
      - Best for varying content
      - Adapts to changes in video characteristics
      
      .. code-block:: bash
         
         cuthrough compress input.mp4 output.mp4 \
             --optimization-strategy ADAPTIVE \
             --quality 0.9 \
             --speed 0.7

   .. py:attribute:: WEIGHTED

      Weighted optimization of multiple metrics.
      
      - Good for specific requirements
      - Balances conflicting goals
      
      .. code-block:: bash
         
         cuthrough compress input.mp4 output.mp4 \
             --optimization-strategy WEIGHTED \
             --quality 0.9 \
             --speed 0.5 \
             --size-reduction 0.7

.. _target-metrics:

Target Metrics
~~~~~~~~~~~~~

.. py:class:: CompressionMetric

   .. py:attribute:: QUALITY

      Quality metric (0-1).
      
      - Measured using VMAF or SSIM
      - Higher values prioritize visual quality
      - Affects encoder settings and bitrate
      
      Example: ``--quality 0.9``

   .. py:attribute:: SPEED

      Speed metric (0-1).
      
      - Compression time relative to video duration
      - Higher values use faster presets
      - Trades quality for speed
      
      Example: ``--speed 0.8``

   .. py:attribute:: SIZE_REDUCTION

      Size reduction metric (0-1).
      
      - Output size relative to input
      - Higher values target smaller files
      - Affects bitrate and compression level
      
      Example: ``--size-reduction 0.7``

.. _compression-profiles:

Compression Profiles
------------------

.. _predefined-profiles:

Predefined Profiles
~~~~~~~~~~~~~~~~~

1. High Quality
^^^^^^^^^^^^^^

.. code-block:: json

   {
     "name": "high_quality",
     "video_codec": "libx264",
     "audio_codec": "aac",
     "container_format": "mp4",
     "video_bitrate": "5M",
     "audio_bitrate": "192k",
     "preset": "veryslow",
     "quality_value": 18,
     "multipass": true
   }

2. Balanced
^^^^^^^^^^

.. code-block:: json

   {
     "name": "balanced",
     "video_codec": "libx264",
     "audio_codec": "aac",
     "container_format": "mp4",
     "video_bitrate": "2M",
     "audio_bitrate": "128k",
     "preset": "medium",
     "quality_value": 23,
     "multipass": false
   }

3. Fast
^^^^^^

.. code-block:: json

   {
     "name": "fast",
     "video_codec": "libx264",
     "audio_codec": "aac",
     "container_format": "mp4",
     "video_bitrate": "1M",
     "audio_bitrate": "128k",
     "preset": "ultrafast",
     "quality_value": 28,
     "multipass": false
   }

.. _custom-profiles:

Custom Profiles
~~~~~~~~~~~~~

Create custom profiles for specific needs:

.. code-block:: bash

   # 4K High Quality Profile
   cuthrough profile create 4k_quality.json \
       --name "4K Quality" \
       --video-codec libx265 \
       --preset slow \
       --video-bitrate 8M \
       --quality-value 20 \
       --multipass

   # Web Optimized Profile
   cuthrough profile create web.json \
       --name "Web Optimized" \
       --video-codec libvpx-vp9 \
       --container webm \
       --video-bitrate 1M \
       --quality-value 31

   # Quick Preview Profile
   cuthrough profile create preview.json \
       --name "Preview" \
       --video-codec libx264 \
       --preset ultrafast \
       --video-bitrate 500k \
       --quality-value 35

.. _quality-assessment:

Quality Assessment
----------------

VMAF (Video Multimethod Assessment Fusion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   VMAF is Netflix's open-source quality metric.

- Perceptual video quality assessment
- Machine learning-based
- More accurate for modern content

SSIM (Structural Similarity Index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   SSIM is used as a fallback when VMAF is unavailable.

- Traditional quality metric
- Measures structural similarity
- Fast computation

Custom Quality Scores
~~~~~~~~~~~~~~~~~~

- Normalized between 0 and 1
- Combines multiple metrics
- Weighted based on content type
- Adapts to user preferences

.. _performance-monitoring:

Performance Monitoring
-------------------

Real-time Metrics
~~~~~~~~~~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4 \
       --save-metrics metrics.json

Example metrics output:

.. code-block:: json

   {
     "metrics": {
       "quality": 0.95,
       "speed": 0.75,
       "size_reduction": 0.65
     },
     "performance": {
       "compression_time": 120.5,
       "cpu_usage": 85.2,
       "memory_usage": 512.0
     }
   }

Progress Tracking
~~~~~~~~~~~~~~

.. py:class:: ProgressTracker

   - Real-time progress display
   - ETA calculation
   - Frame processing rate
   - Size estimation

Logging
~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4 \
       --debug \
       --log-file compression.log

Log output includes:

- Parameter adjustments
- Quality measurements
- Performance metrics
- Error handling

.. _best-practices:

Best Practices
------------

1. Quality-Focused Compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4 \
       --quality 0.95 \
       --optimization-strategy ADAPTIVE \
       --multipass

2. Fast Compression
~~~~~~~~~~~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4 \
       --speed 0.9 \
       --optimization-strategy SIMPLE

3. Size-Focused Compression
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4 \
       --size-reduction 0.8 \
       --optimization-strategy WEIGHTED

4. Balanced Compression
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4 \
       --quality 0.8 \
       --speed 0.7 \
       --size-reduction 0.6 \
       --optimization-strategy ADAPTIVE 