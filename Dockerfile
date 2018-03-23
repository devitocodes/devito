FROM ubuntu:xenial

# This DockerFile is looked after by
MAINTAINER Tim Greaves

# Default gcc version to install
ARG gccvers=4.9

# Add the ubuntu-toolchain-r repository
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-ppa-xenial.list

# Import the Launchpad PPA public key
RUN gpg --keyserver keyserver.ubuntu.com --recv 1E9377A2BA9EF27F
RUN gpg --export --armor BA9EF27F | apt-key add -

# Upgrade to the most recent package set
RUN apt-get update
RUN apt-get -y dist-upgrade

# Install gcc/g++
RUN apt-get -y install gcc-$gccvers g++-$gccvers wget

# Set up for Miniconda
RUN chmod 777 /usr/local
ENV PATH /usr/local/miniconda/bin:$PATH


