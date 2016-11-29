cp software/multicut_src/*.py ${PREFIX}/lib/python2.7/site-packages/
mkdir ${PREFIX}/scripts
cp software/scripts/*.py ${PREFIX}/scripts/.
cp software/scripts/*.sh ${PREFIX}/.
cp software/scripts/README.txt ${PREFIX}/.
chmod a+x ${PREFIX}/run_mc_2d.sh
chmod a+x ${PREFIX}/run_mc_isotropic.sh
chmod a+x ${PREFIX}/run_mc_anisotropic.sh


