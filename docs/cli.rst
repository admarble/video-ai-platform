Command Line Interface
====================

Cuthrough provides a powerful command-line interface with three main commands: ``compress``, ``analyze``, and ``profile``.

Compression Command
-----------------

The ``compress`` command is used to compress video files with various optimization options.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

   cuthrough compress input.mp4 output.mp4

   # With quality settings
   cuthrough compress input.mp4 output.mp4 \
       --quality 0.9 \
       --speed 0.5 \
       --size-reduction 0.6

   # Using a profile
   cuthrough compress input.mp4 output.mp4 \
       --profile high_quality.json

Options
~~~~~~~

.. option:: --quality FLOAT

   Target quality (0-1), default: 0.8

.. option:: --speed FLOAT

   Target speed (0-1), default: 0.6

.. option:: --size-reduction FLOAT

   Target size reduction (0-1), default: 0.7

.. option:: --profile PATH

   Path to compression profile JSON

.. option:: --optimization-strategy {SIMPLE,GRADIENT,ADAPTIVE,WEIGHTED}

   Optimization strategy to use

.. option:: --learning-rate FLOAT

   Learning rate for optimization

.. option:: --history-weight FLOAT

   Weight decay for historical performance

.. option:: --save-metrics PATH

   Save compression metrics to JSON file

Analysis Command
--------------

The ``analyze`` command provides detailed information about video files.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

   cuthrough analyze input.mp4
   cuthrough analyze input.mp4 --json

Output includes:

- Resolution
- Duration
- Frame rate
- Bitrate
- File size
- Container format
- Audio presence

Profile Management
----------------

The ``profile`` command helps manage compression profiles.

Creating Profiles
~~~~~~~~~~~~~~~

.. code-block:: bash

   cuthrough profile create high_quality.json \
       --name "High Quality" \
       --video-codec libx264 \
       --preset veryslow \
       --quality-value 18

Profile Options
~~~~~~~~~~~~~

.. option:: --name NAME

   Profile name

.. option:: --video-codec CODEC

   Video codec (e.g., libx264, libx265, libvpx-vp9)

.. option:: --audio-codec CODEC

   Audio codec (e.g., aac, opus)

.. option:: --container FORMAT

   Container format (e.g., mp4, mkv, webm)

.. option:: --preset PRESET

   Encoder preset (e.g., ultrafast, medium, veryslow)

.. option:: --video-bitrate RATE

   Video bitrate (e.g., 2M, 5M)

.. option:: --audio-bitrate RATE

   Audio bitrate (e.g., 128k, 192k)

.. option:: --quality-value INT

   Quality value (e.g., CRF value)

.. option:: --multipass

   Enable multipass encoding

Common Options
------------

These options are available for all commands:

.. option:: --debug

   Enable debug logging

.. option:: --log-file PATH

   Log to file

.. option:: --quiet

   Suppress progress output

Examples
-------

Basic Compression
~~~~~~~~~~~~~~

.. code-block:: bash

   # Quick compression
   cuthrough compress input.mp4 output.mp4

   # High-quality compression
   cuthrough compress input.mp4 output.mp4 \
       --quality 0.95 \
       --optimization-strategy ADAPTIVE

Advanced Usage
~~~~~~~~~~~~

.. code-block:: bash

   # Create a custom profile
   cuthrough profile create custom.json \
       --name "Custom 4K" \
       --video-codec libx265 \
       --preset slow \
       --video-bitrate 8M \
       --multipass

   # Analyze input video
   cuthrough analyze input.mp4 --json > analysis.json

   # Compress with custom profile and save metrics
   cuthrough compress input.mp4 output.mp4 \
       --profile custom.json \
       --save-metrics metrics.json \
       --debug \
       --log-file compression.log 