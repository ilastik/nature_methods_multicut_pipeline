Build command
-------------

Use the following commands to build this package for the purposes of using it in the multicut pipeline:
   
    $ cd nature_methods_multicut_pipeline/build
    $ export CPLEX_ROOT_DIR='/Users/bergs/Applications/IBM/ILOG/CPLEX_Studio1251'
    $ export WITH_CPLEX=1
    $ conda build --numpy=1.9 opengm-thorsten

For a more recent version of our opengm package build scripts, see:
https://github.com/ilastik/ilastik-build-conda/tree/master/opengm/
