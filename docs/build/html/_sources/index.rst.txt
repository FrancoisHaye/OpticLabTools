.. OpticLabTools documentation master file, created by
   sphinx-quickstart on Wed Jul 22 16:06:31 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================================
Welcome to the OpticLabTools documentation
==========================================

**OpticLabTools** is a a Python library for experimental optics researchers mainly designed for two usages: the study of gaussian beams, and the manipulation of Acousto-Optic Modulators. It was developed during the author's internship in LENS at the University of Florence, for the design of an optical tweezer's like setup. It is comprised of two main packages *ThorlabsGaussianTools* for in real time imaging and fitting of gaussian beams and *MogrfTools* for interfacing the Moglabs XRF synthetizer for the AOM usage and creating arbitrary movements with a set of two perpendicular AOMs.

.. note::
   
   This project is under active development.

.. warning::

   This project has only been tested on very specific conditions: on a windows 11 machine, with a Thorlabs Zelux monochromatic camera and the Moglabs RF synthetizer. Please report any bugs on the github page.

Installation
------------

To use OpticLabTools, first install the wheel archive available in the *dist* directory (*opticlabtools-0.0.1-py3-none-any.whl*) using pip:

.. code-block:: console
   :caption: bash
   
   (.venv) $ pip install opticlabtools-0.0.1-py3-none-any.whl


Contents
--------

.. toctree::
   :maxdepth: 2

   CLI_tools
   ThorlabsGaussianTools
   MogrfTools