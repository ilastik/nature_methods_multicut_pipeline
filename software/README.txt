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

multicur_exp:
    code for running the experiments for the publication on data from
    the ISBI2012 Challenge (http://brainiac2.mit.edu/isbi_challenge/home),
    the SNEMI3D Challenge (http://brainiac2.mit.edu/SNEMI3D/home)
    and the Neuroproof Examples Data (https://github.com/janelia-flyem/neuroproof_examples)

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
