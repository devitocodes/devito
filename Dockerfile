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

# Needed for the conda install later; common across builds so put before gcc to improve cacheing
RUN wget bzip

# Default gcc version to install; can be overridden in Jenkinsfile
ARG gccvers=4.9
# Install gcc/g++
RUN apt-get -y install gcc-$gccvers g++-$gccvers

# Set up for Miniconda
WORKDIR /tmp
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /usr/local/miniconda
ENV PATH /usr/local/miniconda/bin:$PATH
