#! bin/bash
cython edge_volumes.pyx
gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o edge_volumes.so edge_volumes.c

cython numpy_helper.pyx
gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o numpy_helper.so numpy_helper.c
