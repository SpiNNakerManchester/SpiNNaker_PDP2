
PDP<sup>2</sup>: Cognitive Systems Modelling on SpiNNaker
=========================================================

The human brain contains approximately one hundred billion neurons that can be thought of as a network of information processing elements. The Parallel Distributed Processing (PDP) model of cognition is a framework developed by McClelland, Rumelhart, Hinton and others to try to explain how cognition operates in the brain. Much of the power of the PDP models derives from their learning algorithms.

This repository contains software to implement PDP models of cognitive systems on SpiNNaker, a Parallel Distributed Processor (PDP). In this context, PDP systems are modelled using Artificial Neural Networks (ANNs) that apply backpropagation as their learning algorithm. The ANNs implemented here follow the 'Lens' style, but can be easily adapted to a different style or feature set. For further information about Lens see:

[Lens manual @ CMU](https://ni.cmu.edu/~plaut/Lens/Manual)

[crcox/lens github repository](https://github.com/crcox/lens)

The following publication describes the basic algorithm used in the SpiNNaker implementation:

X Jin, M Luján, MM Khan, LA Plana, AD Rast, SR Welbourne and SB Furber, *Algorithm for Mapping Multilayer BP Networks onto the SpiNNaker
Neuromorphic Hardware*, Ninth International Symposium on Parallel and Distributed Computing, Istanbul, Turkey, 2010, pp. 9-16. doi: 10.1109/ISPDC.2010.10 URL: http://ieeexplore.ieee.org/document/5532476/

Release Info
============
This release was tested with the following python versions installed.

- appdirs==1.4.4
- astroid==2.15.8
- attrs==23.1.0
- certifi==2023.7.22
- charset-normalizer==3.3.0
- contourpy==1.1.1
- coverage==7.3.1
- csa==0.1.12
- cycler==0.12.0
- dill==0.3.7
- ebrains-drive==0.5.1
- exceptiongroup==1.1.3
- execnet==2.0.2
- fonttools==4.43.0
- graphviz==0.20.1
- httpretty==1.1.4
- idna==3.4
- importlib-resources==6.1.0
- iniconfig==2.0.0
- isort==5.12.0
- jsonschema==4.19.1
- jsonschema-specifications==2023.7.1
- kiwisolver==1.4.5
- lazy-object-proxy==1.9.0
- lazyarray==0.5.2
- matplotlib==3.7.3
- mccabe==0.7.0
- mock==5.1.0
- MorphIO==6.6.8
- multiprocess==0.70.15
- neo==0.12.0
- numpy==1.24.4
- opencv-python==4.8.1.78
- packaging==23.2
- pathos==0.3.1
- Pillow==10.0.1
- pkgutil_resolve_name==1.3.10
- platformdirs==3.10.0
- pluggy==1.3.0
- pox==0.3.3
- ppft==1.7.6.7
- py==1.11.0
- pylint==2.17.7
- PyNN==0.12.1
- pyparsing==2.4.7
- pytest==7.4.2
- pytest-cov==4.1.0
- pytest-forked==1.6.0
- pytest-instafail==0.5.0
- pytest-progress==1.2.5
- pytest-timeout==2.1.0
- pytest-xdist==3.3.1
- python-coveralls==2.9.3
- python-dateutil==2.8.2
- PyYAML==6.0.1
- quantities==0.14.1
- referencing==0.30.2
- requests==2.31.0
- rpds-py==0.10.3
- scipy==1.10.1
- six==1.16.0
- spalloc==1!7.1.0
- SpiNNaker-PACMAN==1!7.1.0
- SpiNNakerGraphFrontEnd==1!7.1.0
- SpiNNakerTestBase==1!7.1.0
- SpiNNFrontEndCommon==1!7.1.0
- SpiNNMachine==1!7.1.0
- SpiNNMan==1!7.1.0
- SpiNNUtilities==1!7.1.0
- sPyNNaker==1!7.1.0
- testfixtures==7.2.0
- tomli==2.0.1
- tomlkit==0.12.1
- typing_extensions==4.8.0
- urllib3==2.0.5
- websocket-client==1.6.3
- wrapt==1.15.0
- zipp==3.17.0

Development Platform
--------------------

This software is based on the SpiNNaker Graph-Front-End (GFE) platform. The GFE must be installed to use this software. For further information
about the GFE see:

[GFE introduction](http://spinnakermanchester.github.io/graph_front_end/6.0.0/index.html)

[GFE github repository](https://github.com/SpiNNakerManchester/SpiNNakerGraphFrontEnd)

As with most SpiNNaker software, this repository contains C code --that runs on SpiNNaker-- which implements the actual neural network, and python code --that runs on the host machine-- which manages the distribution of tasks across SpiNNaker cores and sets up the communications network to support inter-core communication. This code is also responsible for the downloading of data to SpiNNaker and the collection of results.

License
-------

This software is licensed under the terms of the GNU General Public License v3.0. 

Contributors
------------

Many people have contributed to the development of the software to model cognitive systems on SpiNNaker, amongst them JV Moy, LA Plana, SR Welbourne, X Jin, S Davidson, AD Rast, S Davis and SB Furber, all associated with The University of Manchester at the time of their contribution.

The development of the project has relied heavily on the work of the [SpiNNaker software contributors](http://spinnakermanchester.github.io/common_pages/6.0.0/LicenseAgreement.html#contributors).

Acknowledgments
---------------

Work on this repository started as part of the project 'PDP-squared: Meaningful PDP language models using parallel distributed processors', conducted in collaboration with researchers from the Department of Psychology at The University of Manchester. The project was supported by EPSRC (the UK Engineering and Physical Sciences Research Council) under grant EP/F03430X/1. Ongoing development has been supported by the EU ICT Flagship Human Brain Project under Grants FP7-604102, H2020-720270, H2020-785907 and H2020-945539. LA Plana has been supported by the RAIN Hub, which is funded by the Industrial Strategy Challenge Fund, part of the government’s modern Industrial Strategy. The fund is delivered by UK Research and Innovation and managed by EPSRC under grant EP/R026084/1.

We gratefully acknowledge these institutions for their support.
