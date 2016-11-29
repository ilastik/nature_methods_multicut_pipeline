Obtaining and linking CPLEX:
============================

If you have CPLEX already installed on your machine, you can link to it to via calling the
install-cplex-shared-libs.sh script:
./install-cplex-shared-libs.sh /path/to/cplex-dir /path/to/multicut-software

If you don't have CPLEX installed yet, you must apply for an academic license, for
details see the section Download IBM CPLEX under
http://ilastik.org/documentation/basics/installation.html

For OSX, please use the "OSX" CPLEX installer, not the "darwin" version.


Obtaining the data:
===================

We have bundled the relevant data for reproducing the SNEMI and ISBI experiments, s.t.
they can be directly used with the scripts provided (see below):

http://files.ilastik.org/NaturePaperDataUpl.zip

After unpacking the data, you should have a folder "NaturePaperDataUpl" with two subfolders 
for the experiment data, we will refer to as "\path\to\NaturePaperDataUpl" below.


Running the scripts:
====================

After unpacking the executables, the directory <multicut_root>
(where this README resides) will contain three scripts for the
different use-cases of the software:

- run_mc_2d.sh:           Reproduce the ISBI experiments or similar.
- run_mc_anisotropic.sh:  Reproduce the SNEMI experiments or similar.
- run_mc_isotropic.sh:    Segment isotropic volume data.


run_mc_2d.sh
------------

You can use this script for anisotropic data with groundtruth containing
membrane vs. non-membrane labeling, as in the ISBI 2012 challenge.

Usage:

./run_mc_2d.sh \path\to\NaturePaperDataUpl\ISBI2012 \path\to\ISBI_output_folder

Make sure that the output folder already exists. After a successful run, it
will contain the files 'multicut_segmentation.tif' (a label map stored as
multi-page TIFF) and 'multicut_labeling.tif' (a boundary map, stored as
multi-page TIFF).

Optional argument:

--use_lifted=True
   Use the lifted multicut instead of the standard multicut. This option
   improves the results, but is deactivated by default, because it takes up
   more RAM and runs considerably longer.


run_mc_anisotropic.sh:
----------------------

Use this script for anisotropic data with instance level groundtruth,
as in the SNEMI3D challenge. Note, that reproducing the SNEMI experiments
takes several hours.

Usage:

./run_mc_anisotropic.sh \path\to\NaturePaperDataUpl\SNEMI3D \path\to\SNEMI_output_folder

Make sure that the output folder already exists.

Optional arguments:

--use_lifted=True
   Use the lifted multicut instead of the standard multicut. This option
   improves the results, but is deactivated by default, because it takes up
   more RAM and runs considerably longer.

--snemi_mode=True
   If set to True, runs experiments with the same settings as our SNEMI3D experiments.
   In particular, this uses the precomputed oversegmentation that was corrected for myelin.
   For testing purposes, you should keep this set to False (the default). The True setting
   may take longer, but reproduces the publication's results for the SNEMI dataset.


run_mc_isotropic.sh
-------------------

Use this script for isotropic volume data with instance level groundtruth.
Note that we do not provide exemplary data in 'NaturePaperDataUpl.zip' for this
use-case.

Usage:

./run_mc_isotropic.sh \path\to\your_data \path\to\output_folder

The input data folder must be structured like the SNEMI3D folder provided by us.
Make sure that the output folder already exists.

Optional arguments:

--use_lifted=True
   Use the lifted multicut instead of the standard multicut. This option
   improves the results, but is deactivated by default, because it takes up
   more RAM and runs considerably longer.


For running any of the scripts above on your own data, you need to store it in the same format
and under the same names as the data in the exemplary folder.


-------------------------------------------------------------------------------------------------------------

To generate probability maps with ICv1, please refer to the relevant documentation in /software/nnet


Development and Packaging
=========================

This software was packaged with the conda package manager.
All dependencies can be downloaded from anaconda.org, with some packages sourced
from the 'ilastik' conda channel.

To install the multicut software (and dependencies) to your own conda environment, use this command:
conda install --name my-env --channel ilastik multicut_pipeline

In fact, this entire tarball can be reproduced with the following commands:

$ conda create --name multicut-software --channel ilastik multicut_pipeline
$ tar -czf multicut-software.tar.gz $(conda info --root)/envs/multicut-software

For build recipes of individual packages, refer to the source repository:
https://github.com/ilastik/nature_methods_multicut_pipeline

Note: Development is recommended on Mac/Linux platforms only, using conda-build v2.
      Reproducing these binary pacakges on Windows requires a special version of conda-build.
