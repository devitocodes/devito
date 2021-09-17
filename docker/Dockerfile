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
    chown app:app /docker-entrypoint.sh && \
    chown app:app /print-defaults && \
    chown app:app /tests && \
    chown app:app /jupyter

# Change to the app user.
USER app
WORKDIR /app

ENV DEVITO_ARCH="gcc"
ENV DEVITO_LANGUAGE="openmp"

EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]
