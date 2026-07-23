imaging-gaussian
================

Synopsis
--------

.. code-block:: console
    :caption: bash

    $ imaging-gaussian [OPTIONS]


Description
-----------

Acquisition in real time of images from the thorlabs camera, with a curve fitting og the 2D gaussian, rendered on a matplotlib figure with data on the light distribution.


Options
-------

--help
    Display usage summary and show all available options.

-v, --verbosity
    | type: ``int``
    | How much information you want the code to gove, between 1 and 4 (default: 2).

--exposure-time
    | type: ``int``
    | The exposure time of the camera in µs. (default: 1)

--magnification       
    | type: ``float``
    | The magnification of the imaging system. (default: 23.0)

--lengthscale       
    | type: ``int``
    | The desired size for the scalebar, in µm.

-z, --zoom-width
    | type: ``int``
    | Half width of the zooming window around the point. If None, no zooming is performed (default: 100)


-n, --frame-number
    | type: ``int``
    | Number of frames to acquire.  (default: 1000)

-ds, --downscale
    | type: ``int``
    | Downscaling order, in pixels. it corresponds to the number of pixels removed of the total image. If 0, no downscaling if performed. (default: 0)

-gs, --gaussian-stdev
    | type: ``int``
    | Standard deviation of the gaussian filter applied before downscaling, in pixels. (default: 3)

--gaussian-fit, --no-gaussian-fit
    | type: ``bool``
    | Whether to perform a heavy curve-fitting of the 2D gaussian, or only a calculation of the center of mass and the standard deviation of the data. (default: gaussian-fit)