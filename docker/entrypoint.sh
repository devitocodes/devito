#!/usr/bin/env bash

export PATH=/venv/bin:$PATH

if [[ "$MPIVER" = "HPCX" ]]; then
   echo "loading HPCX"
   source $HPCSDK_HOME/comm_libs/hpcx/latest/hpcx-init.sh
   hpcx_load
fi

if [[ -z "${DEPLOY_ENV}" ]]; then
    exec "$@"
    ./codecov -t -t ${CODECOV_TOKEN} -F "${DEVITO_ARCH}-${DEVITO-PLATFORM}"
else
    exec "$@"
fi