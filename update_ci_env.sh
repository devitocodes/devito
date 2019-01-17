#!/bin/bash

set -euo pipefail

NEW_CONDA_PREFIX=$(mktemp -d)/env

# Create a new conda environment with the default versions users will get
conda env create -p ${NEW_CONDA_PREFIX} -f environment.yml
conda env export -f ci_environment.yml.tmp -p ${NEW_CONDA_PREFIX}

< ci_environment.yml.tmp awk '
	/^name/ { print "name: devito"; next; }
	/^prefix/ { next; }
	/devito==/ { next; }
	/codepy==/ { print "    - \"git+https://github.com/inducer/codepy@10a014fdb89bdf7542cd144441676436809e3d90\""; next; }
	/gen==/ { print "    - \"git+https://github.com/inducer/cgen@361a4c8590910c989ea1cea90d87cd53a393829b\""; next; }
	/pyrevolve==/ { print "    - \"git+https://github.com/opesci/pyrevolve@7edb358fd4006e3948a077573c057c5edb22436d\""; next; }
	{ print; }' > ci_environment.yml
rm ci_environment.yml.tmp

# Now do the same for pip requirements.txt
source activate ${NEW_CONDA_PREFIX}
pip freeze | awk '
	/-e git+git@github.com:opesci\/devito/ { next; }
	/pyrevolve==/ { print "git+https://github.com/opesci/pyrevolve@7edb358fd4006e3948a077573c057c5edb22436d"; next; }
	/cgen==/ { print "git+https://github.com/inducer/cgen@361a4c8590910c989ea1cea90d87cd53a393829b"; next; }
	/codepy==/ { print "git+https://github.com/inducer/codepy@10a014fdb89bdf7542cd144441676436809e3d90"; next; }
	{ print; }' > ci_requirements.txt

rm -rf ${NEW_CONDA_PREFIX}

