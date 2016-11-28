##
## This build script is parameterized by the following external environment variables:
## - WITH_CPLEX
##    - Build OpenGM with CPLEX enabled
##
## - WITH_EXTERNAL_LIBS
##    - Activate OpenGM's "externalLibs" support.  That is, download the external libs,
##      build them, install them, and tweak the opengm python module so it refers to
##      the external libraries in their final install location.
##
## - PREFIX, PYTHON, CPU_COUNT, etc. (as defined by conda-build)

# Convert '0' to empty (all code below treats non-empty as True)
if [[ "$WITH_CPLEX" == "0" ]]; then
    WITH_CPLEX=""
fi

if [[ "$WITH_EXTERNAL_LIBS" == "0" ]]; then
    WITH_EXTERNAL_LIBS=""
fi

# Platform-specific dylib extension
if [ $(uname) == "Darwin" ]; then
    export DYLIB="dylib"
else
    export DYLIB="so"
fi

# Pre-define special flags, paths, etc. if we're building with CPLEX support.
if [[ "$WITH_CPLEX" == "" ]]; then
    CPLEX_ARGS=""
    LINKER_FLAGS=""
else
	CPLEX_LOCATION_CACHE_FILE="$(conda info --root)/share/cplex-root-dir.path"
	
	if [[ "$CPLEX_ROOT_DIR" == "<UNDEFINED>" || "$CPLEX_ROOT_DIR" == "" ]]; then
	    # Look for CPLEX_ROOT_DIR in the cplex-shared cache file.
	    CPLEX_ROOT_DIR=`cat ${CPLEX_LOCATION_CACHE_FILE} 2> /dev/null` \
	    || CPLEX_ROOT_DIR="<UNDEFINED>"
	fi
	
	if [ "$CPLEX_ROOT_DIR" == "<UNDEFINED>" ]; then
	    set +x
	    echo "******************************************"
	    echo "* You must define CPLEX_ROOT_DIR in your *"
	    echo "* environment before building opengm.   *"
	    echo "******************************************"
	    exit 1
	fi

    CPLEX_BIN_DIR=`echo $CPLEX_ROOT_DIR/cplex/bin/x86-64*`
    CPLEX_LIB_DIR=`echo $CPLEX_ROOT_DIR/cplex/lib/x86-64*/static_pic`
    CONCERT_LIB_DIR=`echo $CPLEX_ROOT_DIR/concert/lib/x86-64*/static_pic`
        	
    #LINKER_FLAGS="-L${PREFIX}/lib -L${CPLEX_LIB_DIR} -L${CONCERT_LIB_DIR}"
    #if [ `uname` != "Darwin" ]; then
    #    LINKER_FLAGS="-Wl,-rpath-link,${PREFIX}/lib ${LINKER_FLAGS}"
    #fi

    CPLEX_LIBRARY=${CPLEX_LIB_DIR}/libcplex.${DYLIB}
	CPLEX_ILOCPLEX_LIBRARY=${CPLEX_LIB_DIR}/libilocplex.${DYLIB}
	CPLEX_CONCERT_LIBRARY=${CONCERT_LIB_DIR}/libconcert.${DYLIB}
	
	set +e
	(
	    set -e
	    # Verify the existence of the cplex dylibs.
	    ls ${CPLEX_LIBRARY}
	    ls ${CPLEX_ILOCPLEX_LIBRARY}
	    ls ${CPLEX_CONCERT_LIBRARY}
	)
	if [ $? -ne 0 ]; then
	    set +x
	    echo "************************************************"
	    echo "* Your CPLEX installation does not include     *" 
	    echo "* the necessary shared libraries.              *"
	    echo "*                                              *"
	    echo "* Please install the 'cplex-shared' package:   *"
	    echo "*                                              *"
	    echo "*     $ conda install cplex-shared             *"
	    echo "*                                              *"
	    echo "* (You only need to do this once per machine.) *"
	    echo "************************************************"
	    exit 1
	fi
	set -e

    echo "Building with CPLEX from: ${CPLEX_ROOT_DIR}"
    
    CPLEX_ARGS="-DWITH_CPLEX=ON -DCPLEX_ROOT_DIR=${CPLEX_ROOT_DIR}"
    
    # For some reason, CMake can't find these cache variables on even though we give it CPLEX_ROOT_DIR
    # So here we provide the library paths explicitly
	CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_LIBRARY=${CPLEX_LIBRARY}"
	CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_ILOCPLEX_LIBRARY=${CPLEX_ILOCPLEX_LIBRARY}"
	CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_CONCERT_LIBRARY=${CPLEX_CONCERT_LIBRARY}"
    CPLEX_ARGS="${CPLEX_ARGS} -DCPLEX_BIN_DIR=${CPLEX_CONCERT_LIBRARY}"
fi


##
## START THE BUILD
##

mkdir build
cd build

CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include"
LDFLAGS="${LDFLAGS} -Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"

##
## Download and build external libs
##
EXTERNAL_LIB_FLAGS=""
if [[ "$WITH_EXTERNAL_LIBS" != "" ]]; then
    # We must run cmake preliminarily to enable 'make externalLibs'
	cmake .. \
	        -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
	        -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
	        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.7\
	        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
	        -DCMAKE_PREFIX_PATH=${PREFIX} \
	        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
	        -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
	        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
	        -DCMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS}" \
	        -DCMAKE_CXX_FLAGS_DEBUG="${CXXFLAGS}" \

    make externalLibs

    EXTERNAL_LIB_FLAGS="-DWITH_QBPO=ON -DWITH_PLANARITY=ON -DWITH_BLOSSOM5=ON"
fi

##
## Configure
##
cmake .. \
        -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
        -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.7\
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_PREFIX_PATH=${PREFIX} \
\
        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_DEBUG="${CXXFLAGS}" \
\
        -DBUILD_PYTHON_WRAPPER=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_COMMANDLINE=OFF \
\
        -DWITH_VIGRA=ON \
        -DWITH_BOOST=ON \
        -DWITH_HDF5=ON \
\
        ${EXTERNAL_LIB_FLAGS} \
        ${CPLEX_ARGS} \
##

##
## Compile
##
make -j${CPU_COUNT}

##
## Install to prefix
##
make install

INFERENCE_MODULE_SO=${PREFIX}/lib/python2.7/site-packages/opengm/inference/_inference.so


##
## If we built with external libs, then copy the resulting dylibs to the prefix.
## On Mac, fix the install names.
##
if [[ "$WITH_EXTERNAL_LIBS" != "" ]]; then
   # Install the external dylibs
    mv src/external/libopengm-external-planarity-shared.${DYLIB} "${PREFIX}/lib/"
    mv src/external/libopengm-external-blossom5-shared.${DYLIB} "${PREFIX}/lib/"

    if [ $(uname) == "Darwin" ]; then
        install_name_tool \
            -change \
            $(pwd)/src/external/libopengm-external-planarity-shared.dylib \
            @rpath/libopengm-external-planarity-shared.dylib \
            ${INFERENCE_MODULE_SO}

        install_name_tool \
            -change \
            $(pwd)/src/external/libopengm-external-blossom5-shared.dylib \
            @rpath/libopengm-external-blossom5-shared.dylib \
            ${INFERENCE_MODULE_SO}
    fi
fi

##
## change cplex lib install names.
##
if [[ "$WITH_CPLEX" != "" ]]; then
    (
        if [ `uname` == "Darwin" ]; then
            # Set install names according using @rpath
            install_name_tool -change ${CPLEX_LIB_DIR}/libcplex.dylib     @rpath/libcplex.dylib    ${INFERENCE_MODULE_SO}
            install_name_tool -change ${CPLEX_LIB_DIR}/libilocplex.dylib  @rpath/libilocplex.dylib ${INFERENCE_MODULE_SO}
            install_name_tool -change ${CONCERT_LIB_DIR}/libconcert.dylib @rpath/libconcert.dylib  ${INFERENCE_MODULE_SO}
        fi

#
# ------ SKIP RENAME --------------
#
#        # Rename the opengm package to 'opengm_with_cplex'
#        cd "${PREFIX}/lib/python2.7/site-packages/"
#        mv opengm opengm_with_cplex
#        cd opengm_with_cplex
#        
#        # This sed command works on Mac and Linux
#        for f in $(find . -name "*.py"); do
#	        sed -i.bak 's|import opengm[:space]*$|import opengm_with_cplex|g' "$f"
#	        sed -i.bak 's|from opengm import|from opengm_with_cplex import|g' "$f"
#	        rm "$f.bak"
#        done
    )
fi
