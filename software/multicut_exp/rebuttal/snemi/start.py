import subprocess
import os
import time

def wait4file(f):
    while True:
        if os.path.exists(f):
            break
        time.sleep(10)

def wait4copy(f):
    fsize = 0
    while True:
        file_info = os.stat(f)
        if file_info.st_size > fsize:
            fsize = file_info.st_size
            time.sleep(10)
        else:
            break


if __name__ == '__main__':
    f = "/home/constantin/Work/neurodata_hdd/snemi3d_data/probabilities/pmaps_icv1_train.h5"
    f = "/home/constantin/Work/home_hdd/results/nature_results/rebuttal/snemi/results/snemi_seglmc_icv1_gamma=10.000000.h5"
    #wait4copy(f)
    #wait4file(f)

    #subprocess.call(['python','init_exp.py'])
    subprocess.call(['python','mc_exp.py'])
    subprocess.call(['python','lmc_exp.py'])
