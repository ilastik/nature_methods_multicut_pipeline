#!/bin/bash
source clean_paths.sh
# we assume that this script resides in PREFIX
export PREFIX=$(cd `dirname $0` && pwd)
${PREFIX}/bin/python ${PREFIX}/scripts/run_mc_2d.py "$@"
