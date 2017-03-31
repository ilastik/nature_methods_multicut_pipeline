cython edge_volumes.pyx
gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o edge_volumes.so edge_volumes.c
