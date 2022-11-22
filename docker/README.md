# [Devito] Docker image library

In order to facilitate the dissemination, usage, and development of Devito, we provide a series of Docker images. These images support numerous architectures and compilers and are tagged accordingly. You can find all the available images at [DevitoHub](https://hub.docker.com/r/devitocodes/). The following describes the available images and the workflow to build it yourself. 

## [Devito] images

Devito provides several images that target different architectures and compilers. In the following, all images are described as `imagename-*`. The `*` corresponds to, and should be swapped for, the different release tags `dev`, `latest` or `vx.x.x` depending on if you are interested in specific versions (`vx.x.x`), the latest stable release(`latest`), or the latest development status (`dev`).

### [Devito] on CPU

We provide two CPU images:
- `devito:gcc-*` with the standard GNU gcc compiler.
- `devito:icc-*` with the Intel C compiler for Intel architectures.

These images provide a working environment for any CPU architecture and come with [Devito], `gcc/icc` and `mpi` preinstalled, and utilities such as `jupyter` for usability and exploration of the package.

To run this image locally, you will first need to install `docker`. Then, the following commands will get you started:

```bash
# Pull image and start a bash shell 
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devitocodes/devito:gcc-latest /bin/bash
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm devitocodes/devito:gcc-latest /bin/bash

# or start a Jupyter notebook server on port 8888
docker run --rm -it -p 8888:8888 devitocodes/devito:gcc-latest

```

Alternatively, to run in user context on a cluster with a shared filesystem, you can add the correct user config as `docker` options, e.g.:

```bash
docker run --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devitocodes/devito:gcc-latest python examples/seismic/acoustic/acoustic_example.py
```

**Notes**:
In addition, the following legacy tags are available:

- `devito:cpu-*` that corresponds to `devito:gcc-*`


### [Devito] on GPU

Second, we provide three images to run [Devito] on GPUs, tagged `devito:nvidia-nvc-*`, `devito:nvidia-clang-*`, and `devito:amd-*`.

- `devito:nvidia-nvc-*` is intended to be used on NVidia GPUs. It comes with the configuration to use the `nvc` compiler for `openacc` offloading. This image also comes with CUDA-aware MPI for multi-GPU deployment.
- `devito:nvidia-clang-*` is intended to be used on NVidia GPUs. It comes with the configuration to use the `clang` compiler for `openmp` offloading. This image also comes with CUDA-aware MPI for multi-GPU deployment.
- `devito:amd-*` is intended to be used on AMD GPUs. It comes with the configuration to use the `aoompcc` compiler for `openmp` offloading. This image also comes with ROCm-aware MPI for multi-GPU deployment. This image can also be used on AMD CPUs since the ROCm compilers are preinstalled.

#### NVidia

To run the NVidia GPU version, you will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed and to specify the GPUs to use at runtime with the `--gpus` flag. See, for example, a few runtime commands for the NVidia `nvc` images.


```bash
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devitocodes/devito:nvidia-nvc-latest
docker run --gpus all --rm -it devitocodes/devito:nvidia-nvc-latest python examples/seismic/acoustic/acoustic_example.py

docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm  devitocodes/devito:nvidia-nvc-latest
```

or to run in user context on a cluster with a shared filesystem, you can add the correct user config as `docker` options, e.g.:

```bash
docker run --gpus all --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devitocodes/devito:nvidia-nvc-latest python examples/seismic/acoustic/acoustic_example.py
```


#### AMD

Unlike NVidia, AMD does not require an additional Docker setup and runs with the standard docker. You will, however, need to pass some flags so that the image is linked to the GPU devices. You can find a short walkthrough in these [AMD notes](https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter5/Chapter5.3_%20KerasMultiGPU_ROCm.pdf) for their TensorFlow GPU Docker image.


**Notes**:
In addition, the following legacy tags are available:

- `devito:gpu-*` that corresponds to `devito:nvidia-nvc-*`


## Build a [Devito] image

To build the images yourself, you only need to run the standard build command using the provided Dockerfile. The main difference between the CPU and GPU images will be the base image used.


To build the (default) CPU image, run:

```bash
docker build --network=host --file docker/Dockerfile.devito --tag devito .
```

To build the GPU image with `openacc` offloading and the `nvc` compiler, run:

```bash
docker build --build-arg base=devitocodes/bases:nvidia-nvc --network=host --file docker/Dockerfile.devito --tag devito .
```

or if you wish to use the `clang` compiler with `openmp` offloading:

```bash
docker build --build-arg base=devitocodes/bases:nvidia-clang --network=host --file docker/Dockerfile --tag devito .
```

and finally, for AMD architectures:

```bash
docker build --build-arg base=devitocodes/bases:amd --network=host --file docker/Dockerfile --tag devito .
```


## Debugging a base image

To build the base image yourself locally, you need to run the standard build command using the provided Dockerfile.
Following this, we build the Devito image using the previously built base:

```bash
docker build . --file docker/Dockerfile.cpu --tag devito-gcc --build-arg arch=gcc
docker build . --file docker/Dockerfile.devito --tag devito_img --build-arg base=devito-gcc:latest
```

and then, to run tests or/and examples using the newly built image

```bash
docker run --rm --name testrun devito_img pytest -k "not adjoint" -m "not parallel" tests/
docker run --rm --name testrun devito_img py.test --nbval -k 'not dask' examples/seismic/tutorials/
```

[Devito]:https://github.com/devitocodes/devito

## Developing Devito with Docker

For those aiming to develop in Devito using Docker, you can use docker-compose.
We start by cloning the repo and entering the Devito directory.

```bash
git clone https://github.com/devitocodes/devito.git
cd devito
```

### Example CPU

```bash
# Start a terminal to develop/run for CPUs using docker compose
docker-compose run devito /bin/bash
```

### Example GPU
```bash
# Start a terminal to develop/run for GPUs using docker compose
docker-compose run devito.nvidia /bin/bash
```