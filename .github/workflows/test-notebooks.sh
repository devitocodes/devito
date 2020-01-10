#!/bin/bash
echo "Using compiler: $1"
for notebook in examples/cfd examples/seismic/tutorials examples/compiler examples/userapi
do
    DEVITO_ARCH=$1 pytest --nbval $notebook
done
