# Conda automatically sets these with the -arch x86_64 flag, 
#  which is not recognized by cmake.
export CFLAGS=""
export CXXFLAGS=""
export LDFLAGS=""

if [[ `uname` == 'Darwin' ]]; then
    VIGRA_CXX_FLAGS="-I${PREFIX}/include" # I have no clue why this -I option is necessary on Mac.
    DYLIB_EXT=dylib
else
    VIGRA_CXX_FLAGS="-pthread ${CXXFLAGS}"
    DYLIB_EXT=so
fi

# In release mode, we use -O2 because gcc is known to miscompile certain vigra functionality at the O3 level.
# (This is probably due to inappropriate use of undefined behavior in vigra itself.)
VIGRA_CXX_FLAGS_RELEASE="-O2 -DNDEBUG ${VIGRA_CXX_FLAGS} -std=c++11" 
VIGRA_LDFLAGS="-Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"

# CONFIGURE
mkdir build
cd build


echo $PREFIX


# (Since conda hasn't performed its link step yet, we must 
#  help the tests locate their dependencies via LD_LIBRARY_PATH)
if [[ `uname` == 'Darwin' ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib":"${DYLD_FALLBACK_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$PREFIX/lib":"${LD_LIBRARY_PATH}"
fi

cmake ..\
        -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
        -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_PREFIX_PATH=${PREFIX} \
\
        -DCMAKE_SHARED_LINKER_FLAGS="${VIGRA_LDFLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${VIGRA_LDFLAGS}" \
        -DCMAKE_CXX_FLAGS="${VIGRA_CXX_FLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${VIGRA_CXX_FLAGS_RELEASE}" \
        -DCMAKE_CXX_FLAGS_DEBUG="${VIGRA_CXX_FLAGS}" \
\
        -DWITH_VIGRANUMPY=TRUE \
        -DWITH_BOOST_THREAD=1 \
        -DDEPENDENCY_SEARCH_PREFIX=${PREFIX} \
\
        -DFFTW3F_INCLUDE_DIR=${PREFIX}/include \
        -DFFTW3F_LIBRARY=${PREFIX}/lib/libfftw3f.${DYLIB_EXT} \
        -DFFTW3_INCLUDE_DIR=${PREFIX}/include \
        -DFFTW3_LIBRARY=${PREFIX}/lib/libfftw3.${DYLIB_EXT} \
\
        -DHDF5_CORE_LIBRARY=${PREFIX}/lib/libhdf5.${DYLIB_EXT} \
        -DHDF5_HL_LIBRARY=${PREFIX}/lib/libhdf5_hl.${DYLIB_EXT} \
        -DHDF5_INCLUDE_DIR=${PREFIX}/include \
\
        -DBoost_INCLUDE_DIR=${PREFIX}/include \
        -DBoost_LIBRARY_DIRS=${PREFIX}/lib \
        -DBoost_PYTHON_LIBRARY=${PREFIX}/lib/libboost_python-mt.${DYLIB_EXT} \
        -DBoost_PYTHON_LIBRARY_RELEASE=${PREFIX}/lib/libboost_python-mt.${DYLIB_EXT} \
        -DBoost_PYTHON_LIBRARY_DEBUG=${PREFIX}/lib/libboost_python-mt.${DYLIB_EXT} \
\
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DPYTHON_LIBRARY=${PREFIX}/lib/libpython2.7.${DYLIB_EXT} \
        -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python2.7 \
        -DPYTHON_NUMPY_INCLUDE_DIR=${PREFIX}/lib/python2.7/site-packages/numpy/core/include \
        -DPYTHON_SPHINX=${PREFIX}/bin/sphinx-build \
\
        -DVIGRANUMPY_LIBRARIES="${PREFIX}/lib/libpython2.7.${DYLIB_EXT};${PREFIX}/lib/libboost_python.${DYLIB_EXT};${PREFIX}/lib/libboost_thread.${DYLIB_EXT};${PREFIX}/lib/libboost_system.${DYLIB_EXT}" \
        -DVIGRANUMPY_INSTALL_DIR=${PREFIX}/lib/python2.7/site-packages \
\
        -DZLIB_INCLUDE_DIR=${PREFIX}/include \
        -DZLIB_LIBRARY=${PREFIX}/lib/libz.${DYLIB_EXT} \
\
        -DPNG_LIBRARY=${PREFIX}/lib/libpng.${DYLIB_EXT} \
        -DPNG_PNG_INCLUDE_DIR=${PREFIX}/include \
\
        -DTIFF_LIBRARY=${PREFIX}/lib/libtiff.${DYLIB_EXT} \
        -DTIFF_INCLUDE_DIR=${PREFIX}/include \
\
        -DJPEG_INCLUDE_DIR=${PREFIX}/include \
        -DJPEG_LIBRARY=${PREFIX}/lib/libjpeg.${DYLIB_EXT} \


# BUILD
make -j${CPU_COUNT}

# TEST (before install)
(
    # (Since conda hasn't performed its link step yet, we must 
    #  help the tests locate their dependencies via LD_LIBRARY_PATH)
    if [[ `uname` == 'Darwin' ]]; then
        export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib":"${DYLD_FALLBACK_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="$PREFIX/lib":"${LD_LIBRARY_PATH}"
    fi
    
    # Run the tests
    make -j${CPU_COUNT} check
)

# "install" to the build prefix (conda will relocate these files afterwards)
make install
