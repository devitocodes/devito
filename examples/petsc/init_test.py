import os
from devito.petsc.initialize import PetscInitialize
from devito import configuration
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()
print("helloworld")
