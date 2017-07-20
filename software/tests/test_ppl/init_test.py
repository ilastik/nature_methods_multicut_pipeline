import os
import sys

from multicut_src import DataSet


# TODO read cache folder in as parameter with argparse
data_folder_isotropic = './test_data/isotropic'
meta_folder_isotropic = './cache_isotropic'

data_folder_anisotropic = './test_data/anisotropic'
meta_folder_anisotropic = './cache_anisotropic'


def print_instructions():
    print "Download test data from:"
    print "https://drive.google.com/open?id=0B4_sYa95eLJ1ek8yMWozTzhBbGM"
    print "and extract here to run the tests."
    sys.exit()

def init_ds(data_folder, meta_folder):
    if not os.path.exists(data_folder):
        print_instructions()
    ds = DataSet(meta_folder, 'test')
    ds.add_raw(   os.path.join(data_folder,'raw.h5'),  'data')
    ds.add_input( os.path.join(data_folder,'pmap.h5'), 'data')
    ds.add_seg(   os.path.join(data_folder,'seg.h5'),  'data')
    ds.add_gt(    os.path.join(data_folder,'gt.h5'),   'data')

if __name__ == '__main__':
    init_ds(data_folder_isotropic, meta_folder_isotropic)
    init_ds(data_folder_anisotropic, meta_folder_anisotropic)
