#! /bin/bash

# download the data
wget http://files.ilastik.org/multicut/NaturePaperDataUpl.zip

# unzip
unzip NaturePaperDataUpl.zip
rm NaturePaperDataUpl.zip

cwd=$(pwd)
# make cache directories
mkdir $1
cd $1
mkdir cache_isbi
mkdir cache_snemi
cd $cwd
