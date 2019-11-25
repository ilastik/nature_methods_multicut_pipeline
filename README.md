# nature_methods_multicut_pipeline

This repository contains the software used for the publication [`Multicut brings automated neurite segmentation closer to human performance`](http://www.nature.com/nmeth/journal/v14/n2/full/nmeth.4151.html).
It is a snapshot of the state of the software we developed for this publication.

Since then, we have reimplemented most functionality in an [open-source C++ library](https://github.com/DerThorsten/nifty) and build a [library for more convenient usage of multicut segmentation](https://github.com/constantinpape/elf) as well as a [pipeline for processing large volumes](https://github.com/constantinpape/cluster_tools).
Also, there is an interactive multicut workflow in [ilastik](http://ilastik.org/).

This package (git tag 1.0) reflects the state of the software at the time of publication. Subsequent development happens mainly in the above repositories. 

You can find jupyter notebooks with examples for the core functionality of the pipeline in the examples directory.

If you are interested in using Multicut for your data and cannot make it work with any of the options suggested here, or to get a pointer to the latest post-publication implementation, please do not hesitate to contact us.


## Installation

We provide precompiled binaries at http://files.ilastik.org/multicut/. Just unpack the appropriate file for your operating system (Linux, Windows, or Mac), link against CPLEX and you are ready to go.

To run the multicut solver you need to link against CPLEX:
If you have CPLEX already installed on your machine, you can link to it to via calling the
install-cplex-shared-libs.sh script:
    
    $ ./install-cplex-shared-libs.sh /path/to/cplex-dir /path/to/multicut-binaries

If you don't have CPLEX installed yet, you must apply for an academic license, for
details see the section Download IBM CPLEX under
http://ilastik.org/documentation/basics/installation.html.

Note that you need the academic version to solve any problem of reasonable size, the CPLEX community version is not sufficient.
The file [README.txt](https://github.com/ilastik/nature_methods_multicut_pipeline/blob/1.0/build/multicut_pipeline/TARBALL_README.txt) in the root directory of the unpacked binaries provides detailed usage instructions. 

The multicut software was packaged with [conda](http://conda.pydata.org/docs/). If you are on Linux or Mac and want to modify the code (and have CPLEX installed), you can easily setup your own conda-based development environment (including dependencies) using:

    $ export CPLEX_ROOT_DIR=/path/to/cplex_dir
    $ conda create --name multicut-software --channel ilastik multicut_pipeline

or add the multicut software to an existing environment `my-env` using:

    $ export CPLEX_ROOT_DIR=/path/to/cplex_dir
    $ conda install --name my-env --channel ilastik multicut_pipeline

The tarball of our binary distribution can be reproduced with the commands:

    $ conda create --name multicut-software --channel ilastik multicut_pipeline
    $ tar -czf multicut-software.tar.gz $(conda info --root)/envs/multicut-software

(Note: Development is recommended on Mac/Linux platforms only, using conda-build v2.
       Reproducing these binary packages on Windows requires a special version of conda-build.
       Please contact us if you want to do Windows-based development.)

Note: there are some dependency issues with conda new conda versions >= 4.7.
If you run into these issues, please use conda version 4.6.14, which you can download from [here](https://repo.anaconda.com/miniconda/).
      
The structure of the source code is described in file [`software/README.txt`](https://github.com/ilastik/nature_methods_multicut_pipeline/blob/1.0/software/README.txt) of the source repository at
https://github.com/ilastik/nature_methods_multicut_pipeline. For conda build recipes of individual packages, refer to subdirectory [`build`](https://github.com/ilastik/nature_methods_multicut_pipeline/tree/1.0/build) in that repository.

