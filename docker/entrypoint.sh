#!/usr/bin/env bash

find /app -type f -name '*.pyc' -delete

exec "$@"
