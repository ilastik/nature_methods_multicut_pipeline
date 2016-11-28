Obtaining and linking CPLEX:
============================

If you don't have CPLEX installed yet, you must apply for an academic license, for
details see the section 'Download IBM CPLEX' under
http://ilastik.org/documentation/basics/installation.html

Our software requires CPLEX 12.6.3 (the current version as of 22 April 2016).

When CPLEX is installed, make sure that its binary directory (typically,
"C:\Program Files\ibm\ILOG\CPLEX_Studio1263\cplex\bin\x64_win64") is in the PATH.
For example, open a command window (where you will also run the analysis
scripts later) and type

   set PATH=%PATH%;c:\path\to\cplex


Obtaining the data:
===================

We have bundled the relevant data for reproducing the SNEMI and ISBI experiments,
such that they can be directly used with the scripts provided (see below):

http://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1277965899/NaturePaperDataUpl.zip

After unpacking the data, you should have a folder "NaturePaperDataUpl" which
we will refer to as "c:\path\to\NaturePaperDataUpl" below (the data can reside
on any drive).


Running the scripts:
====================

After unpacking the zipped executables, the directory <multicut_root>
(where this README resides) will contain three scripts for the
different use-cases of the software:

- run_mc_2d.bat:           Reproduce the ISBI experiments or similar.
- run_mc_anisotropic.bat:  Reproduce the SNEMI experiments or similar.
- run_mc_isotropic.bat:    Segment isotropic volume data.


run_mc_2d.bat
-------------

You can use this script for anisotropic data with groundtruth containing
membrane vs. non-membrane labeling, as in the ISBI 2012 challenge.

Usage:

run_mc_2d.bat c:\path\to\NaturePaperDataUpl\ISBI2012 c:\path\to\ISBI_output_folder

Make sure that the output folder already exists. After a successful run, it
will contain the files 'multicut_segmentation.tif' (a label map stored as
multi-page TIFF) and 'multicut_labeling.tif' (a boundary map, stored as
multi-page TIFF).

Optional argument:

--use_lifted=True
   Use the lifted multicut instead of the standard multicut. This option
   improves the results, but is deactivated by default, because it takes up
   more RAM and runs considerably longer.


run_mc_anisotropic.bat:
-----------------------

Use this script for anisotropic data with instance level groundtruth,
as in the SNEMI3D challenge. Note, that reproducing the SNEMI experiments
takes several hours.

Usage:

run_mc_anisotropic.bat c:\path\to\NaturePaperDataUpl\SNEMI3D c:\path\to\SNEMI_output_folder

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


run_mc_isotropic.bat
--------------------

Use this script for isotropic volume data with instance level groundtruth.
Note that we do not provide exemplary data in 'NaturePaperDataUpl.zip' for this
use-case.

Usage:

run_mc_isotropic.bat c:\path\to\your_data c:\path\to\output_folder

The input data folder must be structured like the SNEMI3D folder provided by us.
Make sure that the output folder already exists.

Optional arguments:

--use_lifted=True
   Use the lifted multicut instead of the standard multicut. This option
   improves the results, but is deactivated by default, because it takes up
   more RAM and runs considerably longer.
