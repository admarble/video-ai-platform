Welcome to Cuthrough's documentation!
==================================

Cuthrough is an advanced video compression tool that uses adaptive optimization to achieve the best balance between quality, speed, and file size reduction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   cli
   compression
   api
   contributing

Features
--------

- Adaptive optimization for video compression
- Multiple optimization strategies
- Quality assessment using VMAF/SSIM
- Profile management
- Comprehensive metrics tracking
- Command-line interface

Quick Installation
----------------

.. code-block:: bash

   git clone https://github.com/yourusername/cuthrough.git
   cd cuthrough
   pip install -r requirements.txt
   chmod +x src/cuthrough.py

Basic Usage
----------

.. code-block:: bash

   # Analyze a video
   cuthrough analyze input.mp4

   # Quick compression
   cuthrough compress input.mp4 output.mp4

   # High-quality compression
   cuthrough compress input.mp4 output.mp4 --quality 0.9

For more detailed information, check out the :doc:`quickstart` guide.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 