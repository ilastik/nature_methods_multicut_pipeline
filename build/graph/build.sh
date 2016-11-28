# Conda automatically sets these with the -arch x86_64 flag,
#  which is not recognized by cmake.
export CFLAGS=""
export CXXFLAGS=""
export LDFLAGS=""

# CONFIGURE
mkdir build
cd build

echo ${PREFIX}

if [[ `uname` == 'Darwin' ]]; then
    DYLIB_EXT=dylib
else
    DYLIB_EXT=so
fi

CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include"
LDFLAGS="${LDFLAGS} -Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"

cmake ..\
        -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
        -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_PREFIX_PATH=${PREFIX} \
\
        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_DEBUG="${CXXFLAGS}" \
\
        -DBUILD_PYTHON=ON \
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DPYTHON_LIBRARY=${PREFIX}/lib/libpython2.7.${DYLIB_EXT} \
        -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python2.7 \
        -DPYTHON_INCLUDE_DIR2=${PREFIX}/include \
        -DPYTHON_NUMPY_INCLUDE_DIR=${PREFIX}/lib/python2.7/site-packages/numpy/core/include \
\
        -DBoost_INCLUDE_DIR=${PREFIX}/include \
        -DBoost_LIBRARY_DIRS=${PREFIX}/lib \
        -DBoost_PYTHON_LIBRARY=${PREFIX}/lib/libboost_python-mt.${DYLIB_EXT} \
        -DBoost_PYTHON_LIBRARY_RELEASE=${PREFIX}/lib/libboost_python-mt.${DYLIB_EXT} \
        -DBoost_PYTHON_LIBRARY_DEBUG=${PREFIX}/lib/libboost_python-mt.${DYLIB_EXT} \
\
        -DHDF5_C_INCLUDE_DIR=${PREFIX}/include \
        -DHDF5_hdf5_LIBRARY_DEBUG=${PREFIX}/lib/libhdf5.${DYLIB_EXT} \
        -DHDF5_hdf5_LIBRARY_RELEASE=${PREFIX}/lib/libhdf5.${DYLIB_EXT} \
\
        -DHDF5_CORE_LIBRARY=${PREFIX}/lib/libhdf5.${DYLIB_EXT} \
        -DHDF5_HL_LIBRARY=${PREFIX}/lib/libhdf5_hl.${DYLIB_EXT} \
        -DHDF5_INCLUDE_DIR=${PREFIX}/include \
\
        -DVIGRA_INCLUDE_DIR=${PREFIX}/include \
        -DVIGRA_IMPEX_LIBRARY=${PREFIX}/lib/libvigraimpex.${DYLIB_EXT} \
        -DVIGRA_NUMPY_CORE_LIBRARY=${PREFIX}/lib/python2.7/site-packages/vigra/vigranumpycore.so \
##

make -j ${CPU_COUNT}

# No 'make install' support yet, so copy the files explicitly
graph_pkg_dir=${PREFIX}/lib/python2.7/site-packages/graph
mkdir $graph_pkg_dir
cp python/graph/_graph.so $graph_pkg_dir
cp ../src/andres/graph/python/module/__init__.py $graph_pkg_dir 
