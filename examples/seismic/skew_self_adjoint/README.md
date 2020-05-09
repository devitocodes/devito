# Devito Skew Self Adjoint modeling operators

## These operators are contributed by Chevron Energy Technology Company (2020)

These operators are based on simplfications of the systems presented in:
<br>**Self-adjoint, energy-conserving second-order pseudoacoustic systems for VTI and TTI media for reverse migration and full-waveform inversion** (2016)
<br>Kenneth Bube, John Washbourne, Raymond Ergas, and Tamas Nemeth
<br>SEG Technical Program Expanded Abstracts
<br>https://library.seg.org/doi/10.1190/segam2016-13878451.1

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

## Running unit tests
- if you would like to see stdout when running the tests, use
```py.test -c testUtils.py```

## TODO
<<<<<<< HEAD
- [X] Devito-esque equation version of setup_w_over_q
- [ ] replace the conditional logic in the stencil with comprehension
- [ ] Add memoized methods back to wavesolver.py
- [ ] Add ensureSanityOfFields methods for iso, vti, tti
- [ ] Add timing info via logging for the w_over_q setup, as in initialize_damp
- [ ] Add smoother back to setup_w_over_q method
- [X] Correctness tests
  - [X] Analytic response in the far field
=======
- [ ] Devito-esque equation version of setup_wOverQ
- [ ] figure out if the JacobianAdjointOperator completely solves p0 first
- [ ] figure out if can get time sampling from the SparseTimeFuntions src/rec
- [ ] replace the conditional logic in the stencil with comprehension
- [ ] p --> u in all equations
- [ ] \Gamma --> P_r,P_s
- [ ] Add checkpointing back to the iso wavesolver
- [ ] Farfield similarity tests for correctness, ensure 10 wavelengths out that wavelet phase is preserved
- [ ] Add memoized methods back to wavesolver.py
- [ ] Add ensureSanityOfFields methods for iso, vti, tti
- [ ] Add timing info via logging for the wOverQ setup, as in initialize_damp
- [ ] Add smoother back to setup_WOverQ method
- [ ] Correctness tests
  - [ ] Analytic response in the far field
>>>>>>> 1da4f6b... completed notebook tutorials for SSA isotropic
  - [X] Modeling operator linearity test, with respect to source
  - [X] Modeling operator adjoint test, with respect to source
  - [X] Nonlinear operator linearization test, with respect to model/data
  - [X] Jacobian operator linearity test, with respect to model/data
  - [X] Jacobian operator adjoint test, with respect to model/data
  - [X] Skew symmetry test for shifted derivatives

## To save generated code 

```
f = open("operator.c", "w")
<<<<<<< HEAD
print(op, file=f)
f.close()
```
=======
print(op.ccode, file=f)
f.close()
```

## Some commands for performance testing thread scaling on AMD 7502
```
env OMP_PLACES=cores OMP_PROC_BIND=spread 

env OMP_NUM_THREADS=4  DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.08.txt
env OMP_NUM_THREADS=8  DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.16.txt
env OMP_NUM_THREADS=12 DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.24.txt
env OMP_NUM_THREADS=16 DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.32.txt
env OMP_NUM_THREADS=20 DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.40.txt
env OMP_NUM_THREADS=24 DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.48.txt
env OMP_NUM_THREADS=28 DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.56.txt
env OMP_NUM_THREADS=32 DEVITO_MPI=1 mpirun -n 2 --map-by socket python3 example_iso.py >& mpi.64.txt

env OMP_NUM_THREADS=4  DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.08.txt
env OMP_NUM_THREADS=8  DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.16.txt
env OMP_NUM_THREADS=12 DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.24.txt
env OMP_NUM_THREADS=16 DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.32.txt
env OMP_NUM_THREADS=20 DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.40.txt
env OMP_NUM_THREADS=24 DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.48.txt
env OMP_NUM_THREADS=28 DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.56.txt
env OMP_NUM_THREADS=32 DEVITO_MPI=full mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.64.txt

env OMP_NUM_THREADS=4  DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.08.txt
env OMP_NUM_THREADS=8  DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.16.txt
env OMP_NUM_THREADS=12 DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.24.txt
env OMP_NUM_THREADS=16 DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.32.txt
env OMP_NUM_THREADS=20 DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.40.txt
env OMP_NUM_THREADS=24 DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.48.txt
env OMP_NUM_THREADS=28 DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.56.txt
env OMP_NUM_THREADS=32 DEVITO_MPI=1 OMP_PLACES=cores OMP_PROC_BIND=spread DEVITO_MPI=1 mpirun -n 2 -bind-to socket python3 example_iso.py >& mpi.64.txt

env OMP_NUM_THREADS=32 DEVITO_MPI=1 mpirun -n 2  -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.02.txt
env OMP_NUM_THREADS=16 DEVITO_MPI=1 mpirun -n 4  -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.04.txt
env OMP_NUM_THREADS=10 DEVITO_MPI=1 mpirun -n 6  -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.06.txt
env OMP_NUM_THREADS=8  DEVITO_MPI=1 mpirun -n 8  -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.08.txt
env OMP_NUM_THREADS=6  DEVITO_MPI=1 mpirun -n 10 -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.10.txt
env OMP_NUM_THREADS=5  DEVITO_MPI=1 mpirun -n 12 -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.12.txt
env OMP_NUM_THREADS=4  DEVITO_MPI=1 mpirun -n 14 -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.14.txt
env OMP_NUM_THREADS=4  DEVITO_MPI=1 mpirun -n 16 -prepend-pattern "%r " -bind-to core:16 python3 example_iso.py >& mpi.16.txt

# 7742
env OMP_NUM_THREADS=120 DEVITO_MPI=1 mpirun -n 1  -bind-to core:120 python3 example_iso.py >& mpi.01.txt
env OMP_NUM_THREADS=60  DEVITO_MPI=1 mpirun -n 2  -bind-to core:60  python3 example_iso.py >& mpi.02.txt
env OMP_NUM_THREADS=30  DEVITO_MPI=1 mpirun -n 4  -bind-to core:30  python3 example_iso.py >& mpi.04.txt
env OMP_NUM_THREADS=24  DEVITO_MPI=1 mpirun -n 5  -bind-to core:24  python3 example_iso.py >& mpi.05.txt
env OMP_NUM_THREADS=20  DEVITO_MPI=1 mpirun -n 6  -bind-to core:20  python3 example_iso.py >& mpi.06.txt
env OMP_NUM_THREADS=12  DEVITO_MPI=1 mpirun -n 10 -bind-to core:12  python3 example_iso.py >& mpi.10.txt
env OMP_NUM_THREADS=15  DEVITO_MPI=1 mpirun -n 15 -bind-to core:8   python3 example_iso.py >& mpi.15.txt
env OMP_NUM_THREADS=30  DEVITO_MPI=1 mpirun -n 30 -bind-to core:4   python3 example_iso.py >& mpi.30.txt

# 7502
env OMP_NUM_THREADS=32 DEVITO_MPI=1 mpirun -n 2  -bind-to core:32 python3 example_iso.py >& mpi.02.txt
env OMP_NUM_THREADS=32 DEVITO_MPI=1 mpirun -n 2  -bind-to core:32 python3 example_iso.py >& mpi.02.txt
env OMP_NUM_THREADS=16 DEVITO_MPI=1 mpirun -n 4  -bind-to core:16 python3 example_iso.py >& mpi.04.txt
env OMP_NUM_THREADS=10 DEVITO_MPI=1 mpirun -n 6  -bind-to core:10 python3 example_iso.py >& mpi.06.txt
env OMP_NUM_THREADS=8  DEVITO_MPI=1 mpirun -n 8  -bind-to core:8  python3 example_iso.py >& mpi.08.txt
env OMP_NUM_THREADS=6  DEVITO_MPI=1 mpirun -n 10 -bind-to core:6  python3 example_iso.py >& mpi.10.txt
env OMP_NUM_THREADS=5  DEVITO_MPI=1 mpirun -n 12 -bind-to core:5  python3 example_iso.py >& mpi.12.txt
env OMP_NUM_THREADS=4  DEVITO_MPI=1 mpirun -n 14 -bind-to core:4  python3 example_iso.py >& mpi.14.txt
env OMP_NUM_THREADS=4  DEVITO_MPI=1 mpirun -n 16 -bind-to core:4  python3 example_iso.py >& mpi.16.txt

kill -9 `ps -efla | grep python | awk '{ print $4 }'`
```
>>>>>>> 1da4f6b... completed notebook tutorials for SSA isotropic
