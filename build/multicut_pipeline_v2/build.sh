# Install python modules
mkdir -p ${PREFIX}/multicut_src
cp software/multicut_src/*.py ${PREFIX}/multicut_src
cp -r software/multicut_src/tools ${PREFIX}/multicut_src
cp -r software/multicut_src/false_merges ${PREFIX}/multicut_src
echo "${PREFIX}" > ${PREFIX}/lib/python2.7/site-packages/multicut_src.pth
python -m compileall ${PREFIX}/multicut_src

# Install scripts
mkdir -p ${PREFIX}/scripts
cp software/scripts/*.py ${PREFIX}/scripts/.
cp software/scripts/*.sh ${PREFIX}/.
cp software/scripts/README.txt ${PREFIX}/.
chmod a+x ${PREFIX}/run_mc_2d.sh
chmod a+x ${PREFIX}/run_mc_isotropic.sh
chmod a+x ${PREFIX}/run_mc_anisotropic.sh

# Install README to top-level
cp ${RECIPE_DIR}/TARBALL_README.txt ${PREFIX}/README.txt
