# run mc_2d.sh

# with multicut
python ../scripts/run_mc_2d.py NaturePaperDataUpl/ISBI2012 \
    cache/cache_isbi

echo "#############################"
echo "Test mc2d  on ISBI succesfull"
echo "#############################"

# with lifted multicut
python ../scripts/run_mc_2d.py NaturePaperDataUpl/ISBI2012 \
    cache_isbi \
    --use_lifted true

echo "#############################"
echo "Test lmc2d  on ISBI succesfull"
echo "#############################"


# run mc_anisotropic.sh

# with multicut
python ../scripts/run_mc_anisotropic.py NaturePaperDataUpl/SNEMI3D \
    cache/cache_snemi \
    --snemi_mode true

echo "####################################"
echo "Test mc_aniso  on SNEMI3D succesfull"
echo "####################################"

# with lifted multicut
python run_mc_anisotropic.py NaturePaperDataUpl/SNEMI3D \
    cache/cache_snemi \
    --snemi_mode true \
    --use_lifted true

echo "#####################################"
echo "Test lmc_aniso  on SNEMI3D succesfull"
echo "#####################################"
