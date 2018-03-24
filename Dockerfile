FROM ubuntu:xenial

# This DockerFile is looked after by
MAINTAINER Tim Greaves <tim.greaves@imperial.ac.uk>

# Add the ubuntu-toolchain-r test ppa
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-ppa-xenial.list

# Import the Launchpad PPA public key
RUN gpg --keyserver keyserver.ubuntu.com --recv 1E9377A2BA9EF27F
RUN gpg --export --armor BA9EF27F | apt-key add -

# Upgrade to the most recent package set
RUN apt-get update
RUN apt-get -y dist-upgrade

# Needed for the conda and devito installs later; common across builds so put before gcc to improve cacheing
RUN apt-get -y install wget bzip2 git

# Default gcc version to install; can be overridden in Jenkinsfile
ARG gccvers=4.9
# Install gcc/g++
RUN apt-get -y install gcc-$gccvers g++-$gccvers

# Set up alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$gccvers 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$gccvers 10
RUN update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-$gccvers 10
RUN update-alternatives --install /usr/bin/ar ar /usr/bin/gcc-ar-$gccvers 10
RUN update-alternatives --install /usr/bin/nm nm /usr/bin/gcc-nm-$gccvers 10
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-$gccvers 10
RUN update-alternatives --install /usr/bin/ranlib ranlib /usr/bin/gcc-ranlib-$gccvers 10
RUN update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-$gccvers 10
RUN update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-$gccvers 10

# Set up for Miniconda
WORKDIR /tmp
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /usr/local/miniconda
ENV PATH /usr/local/miniconda/bin:$PATH
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
# Debugging step to finish
RUN conda info -a

# Add working version of devito to image
WORKDIR /usr/local/devito
# Obscure syntax: 'recursively add all the contents of docker's working directory to
#  the working directory in the container
ADD . / ./

# Install devito into the image
RUN conda env create -q -f environment.yml python
RUN source activate devito
RUN pip install -e .
# Debugging step to finish
RUN conda list
