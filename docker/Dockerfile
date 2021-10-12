##############################################################
# This Dockerfile contains the Devito codes.
#
# BUILD:
#   docker build --network=host --file docker/Dockerfile --tag devito .
#
# LEGACY:
#   (1) Option MPI 3.0:
#   docker build --network=host --build-arg MPI_VER=3 --file docker/Dockerfile --tag devito .
#
#   (2) Option MPI 4.0:
#   Enabling and using MPI 4.0.5 works on R450 drivers, but is showing compatibility
#   issues during testing on older R418 drivers.
#
#   docker build --network=host --build-arg MPI_VER=4 --file docker/Dockerfile --tag devito .
#
# RUN:
#   docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devito
#   docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm  devito
#
# to run in user context on a cluster with shared filesystem, you can add the correct user config as docker options e.g.:
#   docker run --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devito python examples/seismic/acoustic/acoustic_example.py
#
##############################################################

FROM python:3.7

RUN apt-get update && apt-get install -y -q \
    mpich \
    libmpich-dev \
    libgl1-mesa-glx

ADD ./requirements.txt /app/requirements.txt
ADD ./requirements-optional.txt /app/requirements-optional.txt
ADD ./requirements-mpi.txt /app/requirements-mpi.txt

RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir jupyter && \
    /venv/bin/pip install --no-cache-dir wheel && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements.txt && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements-optional.txt && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements-mpi.txt && \
    rm -rf ~/.cache/pip

ADD ./devito /app/devito
ADD ./tests /app/tests
ADD ./examples /app/examples
ADD ./benchmarks /app/benchmarks
COPY setup.cfg /app/

ADD docker/run-jupyter.sh /jupyter
ADD docker/run-tests.sh /tests
ADD docker/run-print-defaults.sh /print-defaults
ADD docker/entrypoint.sh /docker-entrypoint.sh

RUN chmod +x \
    /print-defaults \
    /jupyter \
    /tests \
    /docker-entrypoint.sh

## Create App user
# Set the home directory to our app user's home.
ENV HOME=/app
ENV APP_HOME=/app

# Create the home directory for the new app user.
# Create an app user so our program doesn't run as root.
# Chown all the files to the app user.
RUN mkdir -p /app && \
    groupadd -r app && \
    useradd -r -g app -d /app -s /sbin/nologin -c "Docker image user" app && \
    chown -R app:app $APP_HOME && \
    chown -R app:app /docker-entrypoint.sh && \
    chown -R app:app /print-defaults && \
    chown -R app:app /tests && \
    chown -R app:app /jupyter && \
    chown -R app:app /venv

# Change to the app user.
USER app
WORKDIR /app

ENV DEVITO_ARCH="gcc"
ENV DEVITO_LANGUAGE="openmp"

EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]
