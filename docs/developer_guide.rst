Developer Guide
==============

This guide helps developers set up their environment and understand the codebase architecture.

Development Setup
---------------

Prerequisites
~~~~~~~~~~~~

- Python 3.8 or higher
- FFmpeg 4.2+
- Git
- pip and virtualenv

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository and create virtual environment:

   .. code-block:: bash

       git clone https://github.com/yourusername/cuthrough.git
       cd cuthrough
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install development dependencies:

   .. code-block:: bash

       pip install -r requirements-dev.txt

3. Install pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Running Tests
~~~~~~~~~~~

Run the test suite:

.. code-block:: bash

    pytest tests/
    
Run with coverage:

.. code-block:: bash

    pytest --cov=src tests/

System Architecture
-----------------

Component Overview
~~~~~~~~~~~~~~~

Cuthrough consists of several key components:

1. **Core Processor** (``src/cuthrough.py``)
   - Main entry point
   - Orchestrates compression workflow
   - Handles CLI interface

2. **Video Processors** (``src/processors/``)
   - Implements different compression strategies
   - Handles video codec operations
   - Manages quality assessment

3. **Compression Middleware** (``src/compression.py``)
   - Provides compression algorithms
   - Handles streaming operations
   - Manages compression metrics

4. **Security Layer** (``src/security/``)
   - Implements authentication
   - Handles CORS and XSS protection
   - Manages WebSocket security

Component Interaction
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    [CLI/API Entry] → [Core Processor] → [Video Processors]
           ↓                  ↓               ↓
    [Security Layer] ← [Compression Middleware] → [Metrics]

Contributing Guidelines
--------------------

Code Style
~~~~~~~~~

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings in Google style
- Maximum line length: 100 characters

Pull Request Process
~~~~~~~~~~~~~~~~~

1. Create a feature branch:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes and commit:

   .. code-block:: bash

       git commit -m "feat: add your feature description"

3. Write tests for new functionality
4. Update documentation
5. Submit PR against main branch

Commit Message Format
~~~~~~~~~~~~~~~~~~

Follow conventional commits:

- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Test updates
- chore: Maintenance tasks

Example:

.. code-block:: text

    feat(compression): add brotli support
    
    - Implement brotli compression algorithm
    - Add compression level configuration
    - Update tests and documentation

Building Documentation
-------------------

1. Install Sphinx dependencies:

   .. code-block:: bash

       pip install -r docs/requirements.txt

2. Build documentation:

   .. code-block:: bash

       cd docs
       make html

3. View documentation:

   .. code-block:: bash

       open _build/html/index.html  # On macOS
       # On Windows: start _build/html/index.html
       # On Linux: xdg-open _build/html/index.html

Debugging Tips
------------

Using Debug Mode
~~~~~~~~~~~~~

Enable debug logging:

.. code-block:: python

    from cuthrough import logger
    logger.setLevel('DEBUG')

Common debug flags:

.. code-block:: bash

    cuthrough --debug --log-file debug.log compress input.mp4 output.mp4

Using the Debugger
~~~~~~~~~~~~~~~

Example with debugger:

.. code-block:: python

    import pdb

    def process_video(video_path):
        pdb.set_trace()  # Debugger will break here
        # Your code here

Performance Profiling
~~~~~~~~~~~~~~~~~~

Using cProfile:

.. code-block:: bash

    python -m cProfile -o profile.stats your_script.py
    snakeviz profile.stats  # Visualize profile data 