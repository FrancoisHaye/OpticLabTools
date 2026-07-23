CLI tools
=========

Three Command Line Interface (CLI) tools are included in the package to allow easy usage of a Thorlabs camera. It is recommended to create a python virtual environment designed specifically for the OpticLabTools package in order not to encounter problems with other packages.

.. code-block:: console
    :caption: Installation of the packages

    $ python -m venv .venv
    $ .venv\Scripts\activate
    $ pip install opticlabtools.whl

These CLI tools make use at least of a Thorlabs Scientific camera connected to the computer with USB. Installing the Thorlabs dlls should not be necessary, they are included in the package.

The tools provided here aims at making it easier to characterise gaussian beams (which are widely used in optics laboratories) thanks to *real time imaging* of the beam. It was developped during an internship where the author needed to know exactly the waist of the beam when changing the focus of several optical elements. Hence the idea to fit in direct the 2D gaussian beam and see the evolution of the waist when moving optical elements. This could be used for optical tweezers experiments where the size of the collimated beam determines the waist of the focalized one.

.. toctree::
    :caption: Command Line tools

    imaging-simple
    imaging-gaussian
    imaging-calibrate