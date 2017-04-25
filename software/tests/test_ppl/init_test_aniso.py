from multicut_src import DataSet
import os

# TODO upload neuroproof data somewhere and automatically download it
# TODO read cache folder in as parameter with argparse
meta_folder = '/home/constantin/Work/home_hdd/cache/test_cache'
data_folder = '/home/constantin/Work/neurodata_hdd/bock_pedunculus'

def init_test_ds():
    ds = DataSet(meta_folder, 'test')
    ds.add_raw( os.path.join(data_folder,'raw_data.h5'), 'data')
    ds.add_input( os.path.join(data_folder,'rf_prob_map.h5'), 'data')
    ds.add_seg( os.path.join(data_folder,'watershed_dt.h5'), 'data')
    ds.add_gt( os.path.join(data_folder,'gt_mc_bkg.h5'), 'data')

if __name__ == '__main__':
    init_test_ds()
