"""
==============
``MogrfTools``
==============

Package providing necessary utilities to run routines for gaussian defect displacement with a Moglabs RF synthetizer. A Thorlabs scientific camera may also be needed for some functionnalities.

.. currentmodule:: MogrfTools

Classes
=======

.. autosummary::
    :toctree: generated
    :template: custom_class.rst
    :recursive:

    Path - class giving the basic structure for moving the gaussian defect.
    Line - subclass of Path drawing a straigth line.
    Circle - subclass of Path drawing a circle.
    Lissajous - subclass of Path drawing all Lissajous curves.
    MovePoint - class to connect to a mogdevice and launch th defect movement.

Functions
=========

.. autosummary::
    :toctree: generated
    :template: custom_class.rst
    :recursive:

    do_calibration - function to run a heavy calibration experiment with gaussian fitting.
    do_quick_calibration - function to run a lighter calibration experiment without gaussian fitting.
    do_calibration_from_exp - function to analyze results from a calibration experiment previously launched.
    do_complete_calibration_from_exp - function to analyze results from a calibration experiment previously launched, with complete graphs.

"""


try:
    from ThorlabsGaussianTools.utils import configure_path
    configure_path('/thorlabs_dlls/')
except ImportError:
    configure_path = None

from ._mogrf_utilities import *
from ._aom_calibration import *