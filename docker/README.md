# [Devito] docker image library

In order to facilitate the dissemination, usage, and development of Devito, we provide a series of docker images. These images support numerous architectures and compilers and are tagged accordingly. In the following, we describe the available images and the workflow to build it yourself. You can find all the available images at [DevitoHub](https://hub.docker.com/r/devitocodes/). 

## [Devito] images

Devito provides three main images that are targeting different architectures and/or using different compilers. In the following, all images are described as `imagename-*`. The `*` corresponds to, and should be swapped for, the different release tags `dev`, `latest` or `vx.x.x` depending on if you are interested in specific versions (`vx.x.x`), the latest stable release(`latest`), or the latest development status (`dev`)

### [Devito] on CPU

We provide two CPU images:
- `devito:gcc-*` with the standard GNU gcc compiler.
- `devito:icc-*` with the Intel C compiler for Intel architectures.

These base images provide a working [Devito] environment for any CPU architecture and come with [Devito], `gcc/icc` and `mpi` preinstalled as well as utilities such as `jupyter` for usability and exploration of the package.

To run this image locally, you will need `docker` to be installed. Once available, the following commands will get you started:

```bash
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devitocodes/devito:gcc-latest
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm devitocodes/devito:gcc-latest
```

or to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:

```bash
docker run --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devitocodes/devito:gcc-latest python examples/seismic/acoustic/acoustic_example.py
```

**Notes**:
In addition, the following legacy tags are available:

- `devito:cpu-*` that corresponds to `devito:gcc-*`


### [Devito] on GPU

Second, we provide three types of images to run [Devito] on GPUs. These thee images are tagged `devito:nvidia-nvc-*`, `devito:nvidia-clang-*`, and `devito:amd-*`.

- `devito:nvidia-nvc-*` is intended to be used on NVidia GPUs. It comes with the configuration to use the `nvc` compiler for `openacc` offloading. This image also comes with CUDA-aware MPI for multi-GPU deployment.
- `devito:nvidia-clang-*` is intended to be used on NVidia GPUs. It comes with the configuration to use the `clang` compiler for `openmp` offloading. This image also comes with CUDA-aware MPI for multi-GPU deployment.
- `devito:amd-*` is intended to be used on AMD GPUs. It comes with the configuration to use the `aoompcc` compiler for `openmp` offloading. This image also comes with Rocm-aware MPI for multi-GPU deployment. Additionally, this image can be used on AMD CPUs as well since the Rocm compilers are preinstalled. You will need to modify `DEVITO_PLATFORM` to `amd` at runtime to reflect this architecture.

#### NVidia

To run the NVidia GPU version, you will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed and to specify the gpus to use at runtime with the `--gpus` flag. See for example a few runtime commands for the NVidia `nvc` images.


```bash
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devitocodes/devito:nvidia-nvc-latest
docker run --gpus all --rm -it devitocodes/devito:nvidia-nvc-latest python examples/seismic/acoustic/acoustic_example.py

docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm  devitocodes/devito:nvidia-nvc-latest
```

or to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:

```bash
docker run --gpus all --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devitocodes/devito:nvidia-nvc-latest python examples/seismic/acoustic/acoustic_example.py
```


#### AMD

Unlike NVidia, AMD does not require an additional docker setup and runs with the standard docker. You will however need to pass some flags so that the image is linked to the GPU devices. You can find a short walkthrough in these [AMD notes](https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter5/Chapter5.3_%20KerasMultiGPU_ROCm.pdf) for their tensorflow GPU docker image.


**Notes**:
In addition, the following legacy tags are available:

- `devito:gpu-*` that corresponds to `devito:nvidia-nvc-*`


## Build a [Devito] image

To build the images yourself, all you need is to run the standard build command using the provided Dockerfile. The main difference between the CPU and GPU images will be the base image that will be used.


To build the (default) CPU image, simply run:

```bash
docker build --network=host --file docker/Dockerfile.devito --tag devito .
```

And to build the GPU image with `openacc` offloading and the `nvc` compiler, simply run:

```bash
docker build --build-arg base=devitocodes/base:nvidia-nvc --network=host --file docker/Dockerfile.devito --tag devito .
```

or if you wish to use the `clang` compiler with `openmp` offloading:

```bash
docker build --build-arg base=devitocodes/base:nvidia-clang --network=host --file docker/Dockerfile --tag devito .
```

and finally for AMD architectures:

```bash
docker build --build-arg base=devitocodes/base:amd --network=host --file docker/Dockerfile --tag devito .
```


[Devito]:https://github.com/devitocodes/devito