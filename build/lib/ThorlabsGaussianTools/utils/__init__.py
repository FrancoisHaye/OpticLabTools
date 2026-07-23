"""
``utils``
=========

Utility library containing all necessary tools for connection to devices and fitting.

Available functions
-------------------
configure_path
    add the thorlabs dlls directories to the PATH for this python session.

Usefull subpackages
-------------------
tl_camera
    Dialogue API with ThorLabs camera, from ThorLabs Inc.
pygauss
    Gaussian fitting routines
mogdevice
    RF synthetizer API, from MogLabs Inc.

"""

from . import *
from ._windows_setup import configure_path