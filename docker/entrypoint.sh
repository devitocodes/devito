#!/usr/bin/env bash

find /app -type f -name '*.pyc' -delete

export PATH=/venv/bin:$PATH

exec "$@"
