#!/usr/bin/env bash

find /app -type f -name '*.pyc' -delete

export PATH=/venv/bin:$PATH

if [ -f /app/set-devito-env.sh ]; then
    ./app/set-devito-env.sh
fi

exec "$@"
