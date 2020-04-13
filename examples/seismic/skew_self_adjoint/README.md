# Devito Skew Self Adjoint modeling operators

## Tutorial goal

The goal of this series of tutorials is to generate -- and then test for correctness -- the modeling and inversion capability in Devito for variable density visco- acoustics. We use an energy conserving form of the wave equation that is *skew self adjoint*, which allows the same modeling system to be used for all for all phases of finite difference evolution required for quasi-Newton optimization:
- **nonlinear forward**, nonlinear with respect to the model parameters
- **Jacobian forward**, linearized with respect to the model parameters 
- **Jacobian adjoint**, linearized with respect to the model parameters

Pairs of notebooks first implement and then test for correctness with the following three types of physics:

### 1. Variable density visco- acoustic isotropic
- [ssa_01_iso_implementation.ipynb](ssa_01_variable_density_implementation.ipynb)
- [ssa_02_iso_correctness.ipynb](ssa_01_variable_density_correctness.ipynb)

### 2. Variable density pseudo- visco- acoustic VTI anisotropic

### 3. Variable density pseudo- visco- acoustic TTI anisotropic


## These operators are contributed by Chevron Energy Technology Company (2020)

These operators are based on simplfications of the systems presented in:
<br>**Self-adjoint, energy-conserving second-order pseudoacoustic systems for VTI and TTI media for reverse time migration and full-waveform inversion** (2016)
<br>Kenneth Bube, John Washbourne, Raymond Ergas, and Tamas Nemeth
<br>SEG Technical Program Expanded Abstracts
<br>https://library.seg.org/doi/10.1190/segam2016-13878451.1