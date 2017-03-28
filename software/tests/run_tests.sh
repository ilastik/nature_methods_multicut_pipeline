# go to the test directory
cd ../scripts/

# run mc_2d.sh

# with multicut
run_mc_2d.sh --data_folder ../tests/NaturePaperDataUpl/ISBI2012 \
    --output_folder ../tests/cache/cache_isbi

echo "#############################"
echo "Test mc2d  on ISBI succesfull"
echo "#############################"

# with lifted multicut
run_mc_2d.sh --data_folder ../tests/NaturePaperDataUpl/ISBI2012 \
    --output_folder ../tests/cache/cache_isbi \
    --use_lifted true

echo "#############################"
echo "Test lmc2d  on ISBI succesfull"
echo "#############################"


# run mc_anisotropic.sh

# with multicut
run_mc_anisotropic.sh --data_folder ../tests/NaturePaperDataUpl/SNEMI3D \
    --output_folder ../tests/cache/cache_snemi \
    --snemi_mode true

echo "####################################"
echo "Test mc_aniso  on SNEMI3D succesfull"
echo "####################################"

# with lifted multicut
run_mc_anisotropic.sh --data_folder ../tests/NaturePaperDataUpl/SNEMI3D \
    --output_folder ../tests/cache/cache_snemi \
    --snemi_mode true \
    --use_lifted true

echo "#####################################"
echo "Test lmc_aniso  on SNEMI3D succesfull"
echo "#####################################"

cd ../tests/
