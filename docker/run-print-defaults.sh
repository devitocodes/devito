#!/usr/bin/env bash

PYTHONPATH=/app /venv/bin/python -c "
from devito import print_defaults;
print_defaults();
"
