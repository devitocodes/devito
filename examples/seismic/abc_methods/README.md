# Absorbing boundary conditions for the acoustic wave equation  - Applications in seismic problems.

## Authors: Felipe A. G. Silva, Saulo R. M. Barros and Pedro S. Peixoto
    Institute of Mathematics and Statistics - Applied Mathematics Department
   (felipe.augusto.guedes@gmail.com, saulo@ime.usp.br, pedrosp@ime.usp.br)

**Important Informations:** These notebooks are part of the Project Software Technologies for Modeling and Inversion (STMI) at RCGI in the  University of Sao Paulo. 

The objective of these notebooks is to present several schemes which are designed to reduce artificial reflections on boundaries in the numerical solution of the acoustic wave equation with finite differences. We consider several methods, covering absorbing boundary conditions and absorbing boundary layers. Among the schemes, we have implemented:

- Clayton's A1 and A2 boundary conditions;
- Higdon second order boundary conditions;
- Sochaki's type of Damping boundary layer;
- Perfectly Matched Layer (PML);
- Hybrid Absorbing Boundary Conditions (HABC);

The computational implementation of the methods above is done within the framework of <a href="https://www.devitoproject.org/">Devito</a>, which is aimed to produce highly optimized code for finite differences discretizations, generated from high level symbolic problem definitions. Devito presents a work structure in Python and generates code in C ++, which can be taylored for high performance on different computational platforms. The notebooks are organized as follows:

- <a href="01_introduction.ipynb">1. Introduction and description of the acoustic problem;</a>
- <a href="02_damping.ipynb">2. Implementation of Sochaki's damping;</a>
- <a href="03_pml.ipynb">3. PML implementation;</a>
- <a href="04_habc.ipynb">4. HABC (Hybrid absorbing boundary conditions. These encompass also the absorbing boundary conditions A1, A2 and Higdon).;</a>

The notebooks bring a theoretical description of the methods together with the Devito implementation, which can be used for  the simulations of interest. We choose a reference problem, described in the notebook <a href="1_introduction.ipynb">Introduction to Acoustic Problem</a>. The spatial and temporal discretizations used throughout the notebooks are also presented in this introductory notebook, together with other relevant concepts to be used overall. Therefore, one should first assimilate the contents of this notebook. 

In the remaining notebooks, we incrementally describe several numerical techniques to reduce artificial boundary reflections. It is better to follow the order of the notebooks, since concepts are used afterward. We include simulations demonstrating the use of the methods. By changing some parameters, the user would be able to carry out several tests.
