#!/usr/bin/env bash

export PATH=/venv/bin:$PATH

if [[ "$DEVITO_PLATFORM" = "nvidiaX" ]]; then
   echo "loading HPCX"
   source $HPCSDK_HOME/comm_libs/hpcx/latest/hpcx-init.sh
   hpcx_load
fi

if [[ "$DEVITO_ARCH" = "icx" || "$DEVITO_ARCH" = "icc" ]]; then
    echo "Initializing oneapi environement"
    source /opt/intel/oneapi/setvars.sh intel64
fi

if [[ -z "${DEPLOY_ENV}" ]]; then
    exec "$@"
    ./codecov -t ${CODECOV_TOKEN} -F "${DEVITO_ARCH}-${DEVITO-PLATFORM}" || echo "no coverage report exported"
else
    exec "$@"
fi