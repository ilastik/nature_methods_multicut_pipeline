#!/bin/bash
# The run_*.py entry point can be used directly, but this shell
# script cleans the environment to avoid a few potential errors.

# we assume that this script resides in PREFIX
export PREFIX=$(cd `dirname $0` && pwd)

# Do not use the user's previous LD_LIBRARY_PATH settings because they can cause conflicts.
# Start with an empty LD_LIBRARY_PATH
if [[ $LD_LIBRARY_PATH != "" ]]; then
    1>&2 echo "Warning: Ignoring your non-empty LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH=""

# This script is typically used on Linux, because our 
# OSX app uses a different launch mechanism (ilastik.app)
# Still, this script *can* be used on Mac, so let's 
# handle the Mac case, too.
if [[ $DYLD_FALLBACK_LIBRARY_PATH != "" ]] || [[ $DYLD_LIBRARY_PATH != "" ]]; then
    1>&2 echo "Warning: Ignoring your non-empty DYLD_LIBRARY_PATH/DYLD_FALLBACK_LIBRARY_PATH"
fi
export DYLD_LIBRARY_PATH=""
export DYLD_FALLBACK_LIBRARY_PATH=""

# Similarly, clear PYTHONPATH and PYTHONHOME
if [[ $PYTHONPATH != "" ]] || [[ $PYTHONHOME != "" ]]; then
    1>&2 echo "Warning: Ignoring your non-empty PYTHONPATH/PYTHONHOME"
fi    
export PYTHONPATH=""
export PYTHONHOME="${PREFIX}"

# Similarly, disable user-site configuration
export PYTHONNOUSERSITE=1

# Do not use the user's own QT_PLUGIN_PATH, which can cause conflicts with our QT build.
# This is especially important on KDE, which is uses its own version of QT and may conflict.
# Similarly, clear PYTHONPATH and PYTHONHOME
if [[ $QT_PLUGIN_PATH != "" ]]; then
    1>&2 echo "Warning: Ignoring your non-empty QT_PLUGIN_PATH"
fi    
export QT_PLUGIN_PATH=${PREFIX}/plugins

# When Python is compiled with certain (buggy) versions of gcc, 
#  the Python interpreter can sometimes have memory corruption issues 
#  as it shuts down.
# On some systems, memory errors barf out a TON of debug information.
# It's scary that this problem exists, but this output is not useful for users.
# You can disable the checks by uncommenting the following line.
# export MALLOC_CHECK_=0

# fontconfig determines the default paths for configuration files during compile time.
# Make sure to update these to match the local system
export FONTCONFIG_PATH=${PREFIX}/etc/fonts/
export FONTCONFIG_FILE=${PREFIX}/etc/fonts/fonts.conf
