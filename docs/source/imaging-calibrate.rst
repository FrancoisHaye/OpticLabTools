imaging-calibrate
=================

Synopsis
--------

.. code-block:: console
    :caption: bash

    $ imaging-calibrate [OPTIONS]


Description
-----------

Launch the calibration of the AOMs with measurements from the camera. The targets are to determine the movement of the point thanks to the two AOMs of the setup, along with their frequency dependance. Also the intensity is measured and the beam can optionally be fitted with a 2D gaussian function. This is somewhat an obsolete method because intensity measurements with the camera are very poor.
Another performed measurement is the dependance of intensity on the amplitude of the AOMs radio-frequency.


Options
-------

--help
    Display usage summary and show all available options.

-n, --frequencies-number
    | type: ``int``
    | Half number of frequencies on which to perform the calibration (default: 50)

-fi, --f-min
    | type: ``float``
    | Starting frequency, in MHz (default: 50)

-ff, --f-max
    | type: ``float``
    | End frequency, in MHz (default: 110)

-ai, --amplitude-min
    | type : ``float``
    | Start amplitude, in dBm (default: 5)

-af, --amplitude-max
    | type: ``float``
    | End amplitude, in dBm (default: 30)

-l, --lam
    | type: ``float``
    | Lagrande multiplier for spline smoothing of the results. (default: 100)

-fwi, --fw-min
    | type: ``float``
    | Minimal frequency for the optimization window, in MHz. (default: 70)

-fwf, --fw--max
    | type: ``float``
    | Maximal frequency for the optimization window, in MHz. (default: 90)

-nd, --noise-duration
    | type: ``float``
    | Duration of the noise measurement, in s. (default: 60)

-ns, --noise-step
    | type: ``float``
    | Timestep for the noise measurement, in s. (default: 0.5)