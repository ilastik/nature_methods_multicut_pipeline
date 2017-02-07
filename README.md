# nature_methods_multicut_pipeline

This repository contains the software used for the publication about the Multicut pipeline in Nature Methods (http://www.nature.com/nmeth/journal/v14/n2/full/nmeth.4151.html).
It is a snapshot of the state of the software we developed for this publication.

Since then, we have reimplemented some functionality in an open-source C++ pipeline (https://github.com/DerThorsten/nifty) and reimplemented the pipeline code for processing large volumes (https://github.com/constantinpape/McLuigi).
Also, there is a beta version of an interactive multicut workflow in the GUI-based ilastik program (http://ilastik.org/).

This package (git tag 1.0) reflects the state of the software at the time of publication. Subsequent development happens mainly in the above repositories. 

You can find jupyter notebooks with examples for the core functionality of the pipeline in the examples directory.

If you are interested in using Multicut for your data and cannot make it work with any of the options suggested here, or to get a pointer to the latest post-publication implementation, please do not hesitate to contact us.


## Installation

We provide precompiled binaries at http://files.ilastik.org/multicut/. Just unpack the appropriate file for your operating system (Linux, Windows, or Mac), and you are ready to go. The file [README.txt](https://github.com/ilastik/nature_methods_multicut_pipeline/blob/1.0/build/multicut_pipeline/TARBALL_README.txt) in the root directory of the unpacked binaries provides detailed usage instructions. 

The multicut software was packaged with [conda](http://conda.pydata.org/docs/). If you are on Linux or Mac and want to modify the code, you can easily setup your own conda-based development environment (including dependencies) using:

    $ conda create --name multicut-software --channel ilastik multicut_pipeline

or add the multicut software to an existing environment `my-env` using:

    $ conda install --name my-env --channel ilastik multicut_pipeline

In fact, the tarball of our binary distribution can be reproduced with the commands:

    $ conda create --name multicut-software --channel ilastik multicut_pipeline
    $ tar -czf multicut-software.tar.gz $(conda info --root)/envs/multicut-software

(Note: Development is recommended on Mac/Linux platforms only, using conda-build v2.
       Reproducing these binary packages on Windows requires a special version of conda-build.
       Please contact us if you want to do Windows-based development.)
      
The structure of the source code is described in file [`software/README.txt`](https://github.com/ilastik/nature_methods_multicut_pipeline/blob/1.0/software/README.txt) of the source repository at
https://github.com/ilastik/nature_methods_multicut_pipeline. For conda build recipes of individual packages, refer to subdirectory [`build`](https://github.com/ilastik/nature_methods_multicut_pipeline/tree/1.0/build) in that repository.

