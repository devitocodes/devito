#!/bin/bash

# Clean your notebooks!

# Setting this ensures numpy prints scalars rather than np.float() in cell
# output
export PYTEST_VERSION=1

# Clear all the output from all cells
jupyter-nbconvert --clear-output "$@"

# Run the whole notebook in order from start to end
## We don't run with --ClearMetadataPreprocessor.enabled=True
## as this is used to tag cells where nbval should skip checking
jupyter-nbconvert \
    --execute \
    --allow-errors \
    --to notebook \
    --inplace \
    "$@"

# Strip superfluous metadata
## NB: may need more extra-keys, these are just the current offenders
nbstripout \
    --keep-output \
    --keep-count \
    --extra-keys='cell.metadata.editable cell.metadata.slideshow' \
    "$@"
