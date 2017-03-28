# download the data
wget http://files.ilastik.org/multicut/NaturePaperDataUpl.zip

# unzip
unzip NaturePaperDataUpl.zip
rm NaturePaperDataUpl.zip

# make cache directories
mkdir cache
cd cache
mkdir cache_isbi
mkdir cache_snemi
cd ..
