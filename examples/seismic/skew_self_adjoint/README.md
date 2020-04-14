# Devito Skew Self Adjoint modeling operators

## Tutorial goal

The goal of this series of tutorials is to generate -- and then test for correctness -- the modeling and inversion capability in Devito for variable density visco- acoustics. We use an energy conserving form of the wave equation that is *skew self adjoint*, which allows the same modeling system to be used for all for all phases of finite difference evolution required for quasi-Newton optimization:
- **nonlinear forward**, nonlinear with respect to the model parameters
- **Jacobian forward**, linearized with respect to the model parameters 
- **Jacobian adjoint**, linearized with respect to the model parameters

These notebooks first implement and then test for correctness for three types of modeling physics.

| Physics         | Goal                          | Notebook                           |
|:----------------|:----------------------------------|:-------------------------------------|
| Isotropic       | Implementation, nonlinear ops | [ssa_01_iso_implementation1.ipynb] |
| Isotropic       | Implementation, Jacobian ops  | [ssa_02_iso_implementation2.ipynb] |
| Isotropic       | Correctness tests             | [ssa_03_iso_correctness.ipynb]     |
|-----------------|-----------------------------------|--------------------------------------|
| VTI Anisotropic | Implementation, nonlinear ops | [ssa_11_vti_implementation1.ipynb] |
| VTI Anisotropic | Implementation, Jacobian ops  | [ssa_12_vti_implementation2.ipynb] |
| VTI Anisotropic | Correctness tests             | [ssa_13_vti_correctness.ipynb]     |
|-----------------|-----------------------------------|--------------------------------------|
| TTI Anisotropic | Implementation, nonlinear ops | [ssa_21_tti_implementation1.ipynb] |
| TTI Anisotropic | Implementation, Jacobian ops  | [ssa_22_tti_implementation2.ipynb] |
| TTI Anisotropic | Correctness tests             | [ssa_23_tti_correctness.ipynb]     |
|:----------------|:----------------------------------|:-------------------------------------|

[ssa_01_iso_implementation1.ipynb]: ssa_01_iso_implementation1.ipynb
[ssa_02_iso_implementation2.ipynb]: ssa_02_iso_implementation2.ipynb
[ssa_03_iso_correctness.ipynb]:     ssa_03_iso_correctness.ipynb
[ssa_11_vti_implementation1.ipynb]: ssa_11_vti_implementation1.ipynb
[ssa_12_vti_implementation2.ipynb]: ssa_12_vti_implementation2.ipynb
[ssa_13_vti_correctness.ipynb]:     ssa_13_vti_correctness.ipynb
[ssa_21_tti_implementation1.ipynb]: ssa_21_tti_implementation1.ipynb
[ssa_22_tti_implementation2.ipynb]: ssa_22_tti_implementation2.ipynb
[ssa_23_tti_correctness.ipynb]:     ssa_23_tti_correctness.ipynb

## These operators are contributed by Chevron Energy Technology Company (2020)

These operators are based on simplfications of the systems presented in:
<br>**Self-adjoint, energy-conserving second-order pseudoacoustic systems for VTI and TTI media for reverse time migration and full-waveform inversion** (2016)
<br>Kenneth Bube, John Washbourne, Raymond Ergas, and Tamas Nemeth
<br>SEG Technical Program Expanded Abstracts
<br>https://library.seg.org/doi/10.1190/segam2016-13878451.1