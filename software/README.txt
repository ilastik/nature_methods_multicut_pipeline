Source code of the Multicut Pipeline
====================================

This directory contains the source code for the parts of the multicut
pipeline that were designed in our lab. Third-party source code
for the scientific Python ecosystem underlying our programs is not
included.

Subdirectories:
---------------

scripts:
    toplevel scripts to execute the three variants of our pipeline
    (2D, anisotropic 3D, isotropic 3D):
    *.sh:  command-line scripts for Linux/Mac
    *.bat: command-line scripts for Windows
    *.py:  toplevel Python programs called by above scripts
    README*: usage instructions

multicut_src:
    core software (set-up and optimization of multicut problems,
    training and prediction of edge potentials in the MC graph)

nnet:
    training and prediction of membrane probability maps using
    neural networks

dependencies/graph-1.6:
    library to construct the graphs defining the multicut problem

dependencies/wsdt:
    library to compute superpixels from membrane probability maps

examples:
    Jupyter notebooks illustrating key aspects of the pipeline
