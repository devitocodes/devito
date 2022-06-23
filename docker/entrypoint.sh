#!/usr/bin/env bash

find /app -type f -name '*.pyc' -delete

export PATH=/venv/bin:$PATH

if [[ -z "${DEPLOY_ENV}" ]]; then
    exec "$@" && ./codecov -t -t ${CODECOV_TOKEN}
else
    exec "$@"
fi
