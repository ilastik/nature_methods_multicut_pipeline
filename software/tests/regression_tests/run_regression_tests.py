import subprocess

# TODO use execfile instead
subprocess.call(['python','./regression_test_isbi.py'])
subprocess.call(['python','./regression_test_snemi.py'])
subprocess.call(['python','./regression_test_nproof.py'])
