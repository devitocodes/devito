# Devito docker image library


In order to facilitate the dissemination, usage, and development of Devito, we provide a serie of docker images. These images support numerous architectures and compiler and are tagged accordingly. In the following we describe the avaialable images and the workflow to build it yourself. You can find all the available images at [DevitoHub](https://hub.docker.com/r/devitocodes/devito/tags). 

## Devito images

Devito provides three main images that are targeting different architectures and/or using different compilers. In the following, all images are desribed as `imagename-*`. The `*` corresponds to, and should be swapped for, the different release tags `dev`, `latest` or `vx.x.x` depending on if you are interested in specific versions (`vx.x.x`), the latest stable release(`latest`), or the latest development status (`dev`)

### Devito on GPU

`devito:cpu-*`  is the base image that will provide a working Devito environment for any CPU architecture. This image comes with devito and `gcc` preinstalled as well as utilities such as `jupyter` for usability and exploration of the package.

To run this image locally, you will need `docker` to be installed. Once available, the following commands will get you started:

```bash
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devito
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm  devito
```

or to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:

```bash
docker run --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devito python examples/seismic/acoustic/acoustic_example.py
```

### Devito on GPU

Second, we provide two typy of images to run Devito on GPUs. These two images are tagged `devito:nvidia-nvc-*` and `devito:nvidia-clang-*`. The first one will use the `nvc` compiler to compiler and run the Devito generated code using the `openacc` language. The second one will instead use `openmp` as a language and the `clang` compiler for offloading. To run the GPU version, you will need [nvidia-docker] installed an specify the gpu to be used at runtime. See for examples a few runtime commands:


```bash
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devito:nvidia
docker run --gpus all --rm -it devito:nvidia python examples/seismic/acoustic/acoustic_example.py

docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm  devito:nvidia 
```

or to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:

```bash
docker run --gpus all --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devito:nvidia python examples/seismic/acoustic/acoustic_example.py
```

**Notes**:
In addition, the following legacy tags are available:

- `devito:gpu-*` that corresponds to `devito:nvidia-nvc-*`

## Build a Devito image

To build the images yourself, all you need is to run the standard build command using the provided Dockerfile. The main difference between the CPU and GPU images will be the base image that will be used.


To build the (default) CPU image, simply run:

```bash
docker build --network=host --file docker/Dockerfile --tag devito .
```

And to build the GPU image with `openacc` offloading and the `nvc` compiler, simply run:

```bash
docker build base=devito:nvidia --network=host --file docker/Dockerfile --tag devito .
```

or if you wish to use the `llvm-15` (clang) compiler with `openmp` offlaoding:

```bash
docker build base=devito:nvidia-clang --network=host --file docker/Dockerfile --tag devito .
```

