SpiNNaker_PDP2: MLPs on SpiNNaker
=================================

This repository contains software to implement artificial neural
networks based on Multi-layer Perceptrons (MLP) on SpiNNaker.
These MLP networks use back-propagation as their training mechanism.

The MLPs implemented here follow the 'Lens' style, but can easily be
adapted to a different style or feature set. For further information
about Lens see:

http://web.stanford.edu/group/mbc/LENSManual/

The following publication describes the basic algorithm used to implement
MLPs on SpiNNaker:

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

http://spinnakermanchester.github.io/graph_front_end/3.0.0/index.html

https://github.com/SpiNNakerManchester/SpiNNakerGraphFrontEnd

As with most SpiNNaker software, this repository contains C code that
runs on SpiNNaker, which implements the actual MLP network, and python
code that runs on the host machine, which manages the distribution of
tasks across SpiNNaker cores and sets up the communications network
to support inter-core communication. This code is also responsible for
the downloading of data to SpiNNaker and the collection of results.

Acknowledgments
---------------

Work on Multi-layer Perceptrons on SpiNNaker started as part of the
project 'PDP-squared: Meaningful PDP language models using parallel
distributed processors', conducted in collaboration with researchers
from the School of Psychology at The University of Manchester. The project
was supported by EPSRC (the UK Engineering and Physical Sciences Research
Council) under grant EP/F03430X/1. Ongoing development is supported by
the EU ICT Flagship Human Brain Project (FP7-604102). We gratefully
acknowledge these institutions for their support.

Many people have contributed to the development of MLPs on SpiNNaker,
amongst them J Moy, LA Plana, SR Welbourne, X Jin, AD Rast, S Davis
and SB Furber.
