# Effectiveness and computational efficiency of absorbing boundary conditions for full-waveform inversion

## Authors: Daiane Iglesia Dolci, Felipe A. G. Silva, Pedro S. Peixoto and Ernani V. Volpe

## Department of Mechanical Engineering of Polytechnic School, University of São Paulo

## Department of Applied Mathematics, Institute of Mathematics and Statistics, University of São Paulo

## Contacts: dolci@usp.br, felipe.augusto.guedes@usp.br, pedrosp@ime.usp.br, ernvolpe@usp.br

**Important Informations:** These codes are part of the Project Software Technologies for Modeling and Inversion (STMI) at RCGI in the  University of Sao Paulo.

**Acknowledgements:** This research was carried out in association with the ongoing R&D project registered as ANP 20714-2 STMI - Software Technologies for Modelling and Inversion, with applications in seismic imaging (USP/Shell Brasil/ANP).


**Install Instructions:**

0) The Devito Package version that we used in our code is: 4.6+225.

1) Install Devito Package in Linux OS following the instructions below:

1.1) git clone https://github.com/devitocodes/devito.git

1.2) cd devito

1.3) conda env create -f environment-dev.yml

1.4) source activate devito

1.5) pip install -e .

1.6) Install with "conda install" or "pip install" important libreries like: jupyter-notebook, Matplotlib, SEGYIO.

Observation: More datails about install can be found in: https://www.devitoproject.org/devito/download.html

2) To use the FWI code following the instructions below:

2.1) Download the directory "paper-dispersion-reduction" in a desired local.

2.2) Considering the Linux S0, execute the step 1.4).

2.3) Using the know command "python name_file.py" execute the file of interest.

2.4) To acess the notebook execute the comand "jupyter notebook" and navegate into brownser.
