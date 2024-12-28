Internal Documentation
=====================

This section contains documentation intended for internal use by system administrators, DevOps engineers, and developers.

.. warning::
   This documentation contains sensitive information about internal systems and should not be exposed publicly.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core/dead_letter_queue

Infrastructure Components
-----------------------

Dead Letter Queue System
~~~~~~~~~~~~~~~~~~~~~~~
The Dead Letter Queue (DLQ) system is a critical internal component that handles failed task management and retry mechanisms. It provides:

- Reliable error handling for failed video processing tasks
- Configurable retry policies
- Task tracking and monitoring
- Automatic cleanup of old failed tasks

For detailed information, see :doc:`core/dead_letter_queue`. 