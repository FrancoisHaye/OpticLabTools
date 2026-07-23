imaging-simple
==============

Synopsis
--------

.. code-block:: console
    :caption: bash

    $ imaging-simple [OPTIONS]


Description
-----------

Acquisition in real time of the thorlabs camera, rendered on a matplotlib figure.


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