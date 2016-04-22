# Code generation for fast inversion
This is meant to be a plugin for [this project](https://github.com/opesci/inversion) to be able to generate C code for the long-running sections of the inversion workflow. While it might be branched off as a self-sufficient project in the future, for now it is meant to work only within this inversion workflow. 

The intended workflow so far is to symlink this repo as a subdirectory inside the inversion directory.
## Quickstart
1. Add the inversion outer loop python directory to PYTHONPATH
```
cd $INVERSION_OUTER_LOOP_HOME
export PYTHONPATH=$PYTHONPATH:$PWD/python
```
2. Run benchmark
```
cd $CODEGEN_HOME
python benchmark.py
```