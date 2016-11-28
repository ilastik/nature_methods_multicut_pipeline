Obtaining and linking CPLEX:

If you have CPLEX already installed on your machine, you can link to it to via calling the
install-cplex-shared-libs.sh script:
./install-cplex-shared-libs.sh /path/to/cplex-dir /path/to/multicut-binaries

If you don't have CPLEX installed yet, you must apply for an academic license, for
details see the section Download IBM CPLEX under
http://ilastik.org/documentation/basics/installation.html



Obtaining the data:

We have bundled the relevant data for reproducing the SNEMI and ISBI experiments, s.t.
they can be directly used with the scripts provided (see below):

http://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1277965899/NaturePaperDataUpl.zip



Running the scripts:

We provide three scripts for the different use-cases of the software:


- run_mc_2d.sh: 
Reproduce the ISBI experiments. 
Use this script for anisotropic data with groundtruth containing membrane vs. non-membrane labeling.

Usage:
./run_mc_2d.sh /path/to/NaturePaperDataUpl/ISBI2012 /path/to/OutputFolder

Optional arguments:

--use_lifted True
Use the lifted multicut instead of the simple multicut.
This is deactivated by default, because it takes up more RAM and runs considerably longer.


- run_mc_anisotropic.sh:
Reproduce the SNEMI experiments. 
Use this script for anisotropic data with instance level groundtruth.
Note, that reproducing the SNEMI experiments takes several hours.

Usage:
./run_mc_anisotropic.sh /path/to/NaturePaperDataUpl/SNEMI3D /path/to/OutputFolder

Optional arguments:

--use_lifted True
Use the lifted multicut instead of the simple multicut.
This is deactivated by default, because it takes up more RAM and runs considerably longer.

--snemi_mode True
If set to True, runs experiments with the same settings as SNEMI3D experiments. 
In particular, this uses the precomputed oversegmentation that was corrected for myelin.
For test purposes, you should keep this set to False, the True setting may take longer,
but reproduces the publications results for the SNEMI dataset.
Set to False by default.


- run_mc_isotropic.sh:
Use this script for isotropic data with instance level groundtruth.
For this, no exemplary data is provided.

Usage:
./run_mc_isotropic.sh /path/to/YourData /path/to/OutputFolder

Optional arguments:

--use_lifted True
Use the lifted multicut instead of the simple multicut.
This is deactivated by default, because it takes up more RAM and runs considerably longer.

For running any of these scripts on your own data, you need to save it in the same
way as in the folders NaturePaperDataUpl/SNEMI3d or NaturePaperDataUpl/SNEMI3d.

-------------------------------------------------------------------------------------------------------------

To generate probability maps with ICv1, please refer to the relevant documentation in /software/nnet
