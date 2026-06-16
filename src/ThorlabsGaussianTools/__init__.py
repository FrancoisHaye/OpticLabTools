"""
=========================
``ThorlabsGaussianTools``
=========================

Package designed to use a Thorlabs Camera for the analysis of gaussian beams. it is comprised of two main features:

* CLI tools to run an image acquisition directly from the command line;
* Classes for parametrization and launch of image acquisition.

Parameter classes
=================

.. autosummary::
    :toctree: generated/

    CameraParameters - Various parameters needed for camera image acquisition
    VisualizationParameters - Various parameters for matplotlib visualization
    VisualizationGaussianParameters - Sub-class of VisualizationParameters that enriches it for gaussian fitting.

Imaging classes
===============

.. autosummary::
    :toctree: generated/

    RealTimeImaging - abstract class giving the structure of imaging classes and general methods
    SimpleImaging - subclass of RealTimeImaging for imaging without any calculations
    GaussianFitImaging - subclass of RealTimeImaging for realizing a 2D gaussian fit on the image in real time
    RFexpImaging - subclass of GaussianFitImaging for the AOM experiment.

"""

from ._real_time_imaging import *