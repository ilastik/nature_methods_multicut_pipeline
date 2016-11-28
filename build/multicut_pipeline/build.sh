
PARENT_DIR="$(dirname "$(dirname "$RECIPE_DIR")")"
cp $PARENT_DIR/software/multicut_src/*.py ${PREFIX}/lib/python2.7/site-packages/
mkdir $PREFIX/scripts
cp $PARENT_DIR/software/scripts/*.py ${PREFIX}/scripts/.
cp $PARENT_DIR/software/scripts/*.sh ${PREFIX}/.
cp $PARENT_DIR/software/scripts/README.txt ${PREFIX}/.
chmod a+x ${PREFIX}/run_mc_2d.sh
chmod a+x ${PREFIX}/run_mc_isotropic.sh
chmod a+x ${PREFIX}/run_mc_anisotropic.sh


