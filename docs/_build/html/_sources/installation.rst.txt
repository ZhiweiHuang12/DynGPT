.. DynGPT documentation master file, created by
   sphinx-quickstart on Tue Jan 14 16:22:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
---------------

Please follow the guide below to install dyngpt and its dependent
software.


System requirements
~~~~~~~~~~~~~~~~~~~~

DynGPT was developed and tested on Linux (U) and Windows
operating systems. Linux is recommended.

Python requirements
~~~~~~~~~~~~~~~~~~~~~~~~

DynGPT was developed and tested using python 3.8. We recommend using
conda to manage Python environments.

Installation using ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We suggest setting up DynGPT in a separate ``conda``
environment to prevent conflicts with other software dependencies.
Create a new Python environment specifically for dyngpt and install
the required libraries within it.

.. code:: bash

   conda create -n dyngpt_env python=3.8 julia=1.7.3 -c conda-forge
   conda activate dyngpt_env

Download the code and use the cd command to navigate to the directory containing the setup.py file

.. code:: bash

   git clone this-repo-url
   cd DynGPT

Run the following command to install the dyngpt package:

.. code:: bash

   pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
   pip install -r requirements.txt
   pip install .

Now you can check if dyngpt can sucessfully import in Python:

.. code:: python

   import dyngpt as dp

DynGPT requires additional Julia packages for stochastic simulation of STNs.  These can be easily installed through
following command:

.. code:: bash

   activate your Julia
   using Pkg
   Pkg.add(["Catalyst", "StaticArrays", "Distributions", "JumpProcesses", "CSV", "DataFrames", "JSON", "StatsBase", "Sobol"])

.. raw:: html

   <h1 style="font-size: 30px;">Troubleshooting</h1>

**Continuous updating…**

Feel free to contact us if you encounter any other problems with the
installation.