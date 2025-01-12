from pathlib import Path
import os

# Create docs directory structure
docs_structure = """
docs/
├── Makefile
├── make.bat
├── source/
    ├── conf.py
    ├── index.rst
    ├── installation.rst
    ├── api/
    │   ├── index.rst
    │   ├── frame_extraction.rst
    │   ├── scene_processor.rst
    │   ├── object_detector.rst
    │   ├── audio_processor.rst
    │   └── text_video_aligner.rst
    ├── modules/
    │   ├── index.rst
    │   ├── video_processing.rst
    │   ├── audio_processing.rst
    │   └── utilities.rst
    └── _static/
"""

# Configuration file (conf.py)
conf_py = """
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Video AI Platform'
copyright = '2024, Video AI Team'
author = 'Video AI Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
"""

# Main index.rst
index_rst = """
Welcome to Video AI Platform's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api/index
   modules/index

Overview
--------
The Video AI Platform is a comprehensive video processing system that provides:

* Frame extraction and analysis
* Scene detection and classification
* Object detection and tracking
* Audio processing and transcription
* Text-video alignment capabilities

Getting Started
-------------
Check out the :doc:`installation` guide to get started with the platform.

API Documentation
---------------
For detailed API documentation, see the :doc:`api/index` section.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

# API index.rst
api_index_rst = """
API Reference
============

.. toctree::
   :maxdepth: 2

   frame_extraction
   scene_processor
   object_detector
   audio_processor
   text_video_aligner
"""

# Frame extraction API doc
frame_extraction_rst = """
Frame Extraction
==============

.. automodule:: src.video_processor
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
------------

.. autofunction:: _extract_frames

Exception Classes
--------------

.. autoclass:: VideoProcessingError
   :members:
   :show-inheritance:
"""

# Installation guide
installation_rst = """
Installation Guide
================

Requirements
-----------
* Python 3.8+
* CUDA-compatible GPU (recommended)
* FFmpeg

Quick Install
-----------

.. code-block:: bash

   git clone https://github.com/admarble/video-ai-platform.git
   cd video-ai-platform
   pip install -r requirements.txt

Development Setup
--------------

For development, install additional dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt

GPU Support
---------

To enable GPU support, ensure you have:

1. NVIDIA GPU drivers installed
2. CUDA Toolkit (11.0+)
3. cuDNN (8.0+)

Verify your installation:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
"""

def setup_sphinx_docs():
    # Create directory structure
    os.makedirs("docs/source/api", exist_ok=True)
    os.makedirs("docs/source/modules", exist_ok=True)
    os.makedirs("docs/source/_static", exist_ok=True)
    
    # Write configuration files
    with open("docs/source/conf.py", "w") as f:
        f.write(conf_py)
        
    with open("docs/source/index.rst", "w") as f:
        f.write(index_rst)
        
    with open("docs/source/api/index.rst", "w") as f:
        f.write(api_index_rst)
        
    with open("docs/source/api/frame_extraction.rst", "w") as f:
        f.write(frame_extraction_rst)
        
    with open("docs/source/installation.rst", "w") as f:
        f.write(installation_rst)

if __name__ == "__main__":
    setup_sphinx_docs() 