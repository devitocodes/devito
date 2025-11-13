#!/bin/bash

pre-commit run --all-files trailing-whitespace
git add --all
git commit --no-verify -m "lint: Remove all the trailing whitespace"

pre-commit run --all-files end-of-file-fixer
git add --all
git commit --no-verify -m "lint: Fix ends of files"

isort .
git add --all
git commit --no-verify -m "lint: Re-sort all imports with new isort rules"

ruff check --fix
git add --all
git commit --no-verify -m "lint: First pass with ruff --fix"

# Don't run these
# ruff check --fix --unsafe-fixes
# git add --all
# git commit --no-verify -m "lint: Second pass with ruff --fix --unsafe-fixes"
