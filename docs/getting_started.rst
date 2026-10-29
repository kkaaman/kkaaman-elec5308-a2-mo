Getting Started
===============

Requirements
------------

Python 3.10 or later is supported.

The CARLA demonstration was developed and tested using Python 3.12 and
CARLA 0.9.16.

Installation
------------

Clone the repository and create a virtual environment.

Linux / WSL:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[test,docs]"

Windows:

.. code-block:: powershell

   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -e ".[test,docs]"

Running the tests
-----------------

From the repository root:

.. code-block:: bash

   python -m pytest

Running the CARLA demos
-----------------------

Start CARLA before running either demonstration.

From ``src/``:

.. code-block:: powershell

   python demo_combined_perception.py --town Town03

or:

.. code-block:: powershell

   python demo_fused_lane_following.py --town Town03

Reproducible offline workflow
-----------------------------

See :doc:`reproduction` for recorded runtime versions, local assets, a disconnected
installation recipe and explicit CARLA/GPU checks. A successful package installation
alone does not establish driving acceptance.
