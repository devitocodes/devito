##############################################################
# This Dockerfile contains the Devito codes and can be buld using different base images.
##############################################################

ARG base=python:3.7
FROM $base

ARG base

# Install for basic base not containing it
RUN if [ "$base" = "python:3.7" ]; then \
      apt-get update && apt-get install -y mpich libmpich-dev libgl1-mesa-glx; \
    fi

##################  Install devito ############################################
# Copy Devito
ADD . /app/devito
# Remove git files
RUN rm -rf /app/devito/.git

# Install pip dependencies and devito as a pip package
RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir jupyter && \
    /venv/bin/pip install --no-cache-dir wheel && \
    /venv/bin/pip install --no-cache-dir -e /app/devito[extras,mpi] && \
    rm -rf ~/.cache/pip

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

# Reset any existin CFLAG
ENV CFLAGS=

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
WORKDIR /app/devito

# Env vars from ARGS
ENV DEVITO_ARCH="gcc"
ENV DEVITO_LANGUAGE="openmp"
ENV DEVITO_LOGGING="INFO"

EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]
