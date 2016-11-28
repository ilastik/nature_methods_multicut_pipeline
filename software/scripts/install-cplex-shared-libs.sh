#
# Given the location of CPLEX static libraries, convert them to shared 
# libraries and install them into the environment lib directory.
#

#
# Check usage
#
if [ $# != 1 ]; then
    echo "Usage: $0 <path-to-CPLEX_Studio1251>" 1>&2
    exit 1
fi

#
# Read args
#
CPLEX_ROOT_DIR="$1"
PREFIX=$(cd $(dirname $0) && pwd)

if [[ ${PREFIX##*/} != "multicut-software" ]]; then
    echo "This script must be located within the multicut-software directory, and cannot be relocated." 1>&2
    exit 1
fi


#
# Validate arg
#
if [ ! -d "$CPLEX_ROOT_DIR/cplex" ]; then
    echo "Error: $CPLEX_ROOT_DIR does not appear to be the CPLEX installation directory." 1>&2
    exit 2
fi

if [ ! -d "$PREFIX/lib" ]; then
    echo "Error: $PREFIX does not appear to be the directory of multicut-software." 1>&2
    exit 2
fi


CPLEX_LIB_DIR=`echo $CPLEX_ROOT_DIR/cplex/lib/x86-64*/static_pic`
CONCERT_LIB_DIR=`echo $CPLEX_ROOT_DIR/concert/lib/x86-64*/static_pic`

#
# Are we using clang?
#
g++ 2>&1 | grep clang > /dev/null
GREP_RESULT=$?
if [ $GREP_RESULT == 0 ]; then
    # Using clang, must specify libstdc++ (not libc++, which is the default).
    STDLIB_ARG="-stdlib=libstdc++"
else
    STDLIB_ARG=""
fi

#
# Create a shared library from each static cplex library,
# and write it directly into ${PREFIX}/lib
#
if [ `uname` == "Darwin" ]; then
    set -x
    g++ -fpic -shared -Wl,-all_load ${CPLEX_LIB_DIR}/libcplex.a     $STDLIB_ARG -o ${PREFIX}/lib/libcplex.dylib    -Wl,-no_compact_unwind -Wl,-install_name,@loader_path/libcplex.dylib
    g++ -fpic -shared -Wl,-all_load ${CONCERT_LIB_DIR}/libconcert.a $STDLIB_ARG -o ${PREFIX}/lib/libconcert.dylib  -Wl,-no_compact_unwind -Wl,-install_name,@loader_path/libconcert.dylib
    g++ -fpic -shared -Wl,-all_load ${CPLEX_LIB_DIR}/libilocplex.a  $STDLIB_ARG -o ${PREFIX}/lib/libilocplex.dylib -Wl,-no_compact_unwind -Wl,-install_name,@loader_path/libilocplex.dylib \
        -L${PREFIX}/lib -lcplex -lconcert
    set +x
else
    set -x
    g++ -fpic -shared -Wl,-whole-archive ${CPLEX_LIB_DIR}/libcplex.a     -Wl,-no-whole-archive -o ${PREFIX}/lib/libcplex.so
    g++ -fpic -shared -Wl,-whole-archive ${CONCERT_LIB_DIR}/libconcert.a -Wl,-no-whole-archive -o ${PREFIX}/lib/libconcert.so
    g++ -fpic -shared -Wl,-whole-archive ${CPLEX_LIB_DIR}/libilocplex.a  -Wl,-no-whole-archive -o ${PREFIX}/lib/libilocplex.so
    set +x
fi

echo "---"
echo "CPLEX shared libraries installed to: ${PREFIX}/lib"
echo "DONE."