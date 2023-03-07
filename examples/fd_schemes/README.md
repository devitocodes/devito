# High Order Finite Difference Schemes for Numerical Dispersion Reduction on Acoustic Waves

## Authors: Felipe A. G. Silva and Pedro S. Peixoto

## Department of Applied Mathematics, Institute of Mathematics and Statistics, University of São Paulo

## Contacts: felipe.augusto.guedes@usp.br and pedrosp@ime.usp.br

**Highlights:** In this directory we have added the codes of the Classical Finite Difference Schemes and the Finite Difference Schemes with Dispersion Reduction Property, in addition to the study that we are developing with Classical Schemes with Extra points.

**Important Informations:** These codes are part of the Project Software Technologies for Modeling and Inversion (STMI) at RCGI in the  University of Sao Paulo.

**Acknowledgements:** This research was carried out in association with the ongoing R&D project registered as ANP 20714-2 STMI - Software Technologies for Modelling and Inversion, with applications in seismic imaging (USP/Shell Brasil/ANP).

**Main Notebook:**

- <a href="new_stencils1.ipynb"> Dispersion Reduction Schemes Applied in Acoustic Wave Problem - Stencil Test;</a>
- <a href="new_stencils2.ipynb"> Dispersion Reduction Schemes Applied in Acoustic Wave Problem - Cross-Line Comparision Results;</a>

**Install Instructions:**

0) The Devito Package version that we used in our code is: 4.7.1+23

1) Install Devito Package in Linux OS following the instructions below:

1.1) git clone https://github.com/devitocodes/devito.git

1.2) cd devito

1.3) conda env create -f environment-dev.yml

1.4) source activate devito

1.5) pip install -e .

1.6) Install with "conda install" or "pip install" important libreries like: jupyter-notebook, Matplotlib, SEGYIO.

Observation: More datails about install can be found in: https://www.devitoproject.org/devito/download.html
