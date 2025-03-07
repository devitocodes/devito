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

if [[ -v CODECOV_TOKEN ]]; then
    echo "CODECOV_TOKEN provided, running and uploading coverage report"
    "$@"
    ./codecov --verbose upload-process --disable-search \
        -t ${CODECOV_TOKEN} \
        -F "pytest-gpu-${DEVITO_ARCH}-${DEVITO_PLATFORM}" \
        -n "pytest-gpu-${DEVITO_ARCH}-${DEVITO_PLATFORM}" \
        -f coverage.xml
else
    exec "$@"
fi