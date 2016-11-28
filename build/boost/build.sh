#!/bin/bash

# Hints:
# http://boost.2283326.n4.nabble.com/how-to-build-boost-with-bzip2-in-non-standard-location-td2661155.html
# http://www.gentoo.org/proj/en/base/amd64/howtos/?part=1&chap=3
# http://www.boost.org/doc/libs/1_55_0/doc/html/bbv2/reference.html

# Hints for OSX:
# http://stackoverflow.com/questions/20108407/how-do-i-compile-boost-for-os-x-64b-platforms-with-stdlibc

# Build dependencies:
# - bzip2-devel

if [ `uname` == Darwin ]; then
    B2ARGS="toolset=darwin"
    echo "using darwin : : ${PREFIX}/bin/g++" > user-config.jam
else
    B2ARGS="toolset=gcc"
fi

mkdir -vp ${PREFIX}/bin;

./bootstrap.sh \
  --with-libraries=date_time,filesystem,python,regex,serialization,system,test,thread,program_options,chrono,atomic,random \
  --with-python=${PYTHON} \
  --prefix=${PREFIX}

# In the commands below, we want to include linkflags=blabla and 
# cxxflags=blabla, but only if there are actual values for 
# linkflags and cxxflags.  Otherwisde, omit those settings entirely.
LINK_ARG=""
if [ "${CXX_LDFLAGS}" != "" ]; then
    LINK_ARG=linkflags=
fi

echo "LINK_ARG=$LINK_ARG"

CXX_ARG=""
if [ "${CXXFLAGS}" != "" ]; then
    CXX_ARG=cxxflags=
fi


# Create with --layout=tagged to create libraries named with -mt convention
./b2 \
  --layout=tagged \
  -j ${CPU_COUNT} \
  -sNO_BZIP2=1 \
  variant=release \
  threading=multi \
  ${B2ARGS} \
  ${CXX_ARG}"${CXXFLAGS}" \
  ${LINK_ARG}"${CXX_LDFLAGS}" \
  install
  
# Add symlinks in case some dependencies expect non-tagged names.
cd ${PREFIX}/lib
for f in libboost_*-mt*; do
    echo $f
    f_without_mt=${f/-mt/}
    ln -s $f $f_without_mt
done

# Omitted these options from above commands:  
#  -sZLIB_INCLUDE=${PREFIX}/include \
#  -sZLIB_SOURCE=${zlib_SRC_DIR} \
