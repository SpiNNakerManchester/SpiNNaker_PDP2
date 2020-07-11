SpiNNaker_PDP2: Back-Propagation ANNs on SpiNNaker
====================================================

This repository contains software to implement artificial neural
networks (ANNs) on SpiNNaker.
These ANNs use back-propagation as their training mechanism.

The ANNs implemented here follow the 'Lens' style, but can easily be
adapted to a different style or feature set. For further information
about Lens see:

[Lens manual @ CMU](https://ni.cmu.edu/~plaut/Lens/Manual)

[crcox/lens github repository](https://github.com/crcox/lens)

The following publication describes the basic algorithm used in the
SpiNNaker implementation:

X Jin, M Luján, MM Khan, LA Plana, AD Rast, SR Welbourne and SB Furber,
*Algorithm for Mapping Multilayer BP Networks onto the SpiNNaker
Neuromorphic Hardware*,
Ninth International Symposium on Parallel and Distributed Computing,
Istanbul, Turkey, 2010, pp. 9-16.
doi: 10.1109/ISPDC.2010.10
URL: http://ieeexplore.ieee.org/document/5532476/

Development Platform
--------------------

This software is based on the SpiNNaker Graph-Front-End (GFE) platform.
The GFE must be installed to use this software. For further information
about the GFE see:

[GFE introduction](http://spinnakermanchester.github.io/graph_front_end/5.0.0/index.html)

[GFE github repository](https://github.com/SpiNNakerManchester/SpiNNakerGraphFrontEnd)

As with most SpiNNaker software, this repository contains C code --that
runs on SpiNNaker-- which implements the actual neural network, and python
code --that runs on the host machine-- which manages the distribution of
tasks across SpiNNaker cores and sets up the communications network
to support inter-core communication. This code is also responsible for
the downloading of data to SpiNNaker and the collection of results.

License
-------

This software is licensed under the terms of the GNU General Public License v3.0. 

Contributors
------------

Many people have contributed to the development of ANNs on SpiNNaker,
amongst them J Moy, LA Plana, SR Welbourne, X Jin, S Davidson, AD Rast,
S Davis and SB Furber, all associated with The University of Manchester
at the time of their contribution.

The development of the project has relied heavily on the work of the
[SpiNNaker software contributors](http://spinnakermanchester.github.io/common_pages/5.0.0/LicenseAgreement.html#contributors).

Acknowledgments
---------------

Work on this repository started as part of the
project 'PDP-squared: Meaningful PDP language models using parallel
distributed processors', conducted in collaboration with researchers
from the Department of Psychology at The University of Manchester. The project
was supported by EPSRC (the UK Engineering and Physical Sciences Research
Council) under grant EP/F03430X/1. Ongoing development has been supported by
the EU ICT Flagship Human Brain Project under Grants FP7-604102, H2020-720270,
H2020-785907 and H2020-945539.
LA Plana is supported by the RAIN Hub, which is
funded by the Industrial Strategy Challenge Fund, part of the government’s
modern Industrial Strategy. The fund is delivered by UK Research and
Innovation and managed by EPSRC under grant EP/R026084/1.
We gratefully acknowledge these institutions for their support.
