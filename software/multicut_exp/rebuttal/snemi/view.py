import vigra

from init_exp import meta
from volumina_viewer import volumina_n_layer

def view_train():
    ds = meta.get_dataset('snemi3d_train')
    pmap = vigra.readHDF5('/home/constantin/Downloads/traininf-cst-inv.h5', 'data')
    volumina_n_layer([ds.inp(0), ds.inp(1), pmap, ds.seg(0),ds.gt()])

def view_test(res1, res2):
    ds = meta.get_dataset('snemi3d_test')
    #volumina_n_layer([ds.inp(0), ds.inp(1), pm_new, pm_new1], ['raw','pm_old', 'pm_new1', 'pm_new2'])
    #else:
    volumina_n_layer([ds.inp(0), ds.inp(1), ds.seg(0), res1, res2], ['raw','pmap','ws','curr_res','best_res'])


def view_test_pmaps(new_pmaps):
    ds = meta.get_dataset('snemi3d_test')

    raw = ds.inp(0)
    pm_old = ds.inp(1)
    pm_2d = vigra.readHDF5('/home/constantin/Work/neurodata_hdd/snemi3d_data/probabilities/pmaps_icv2_test.h5', 'data')
    data = [raw, pm_old, pm_2d]
    data.extend(new_pmaps)
    labels = ['raw', '3d_v2', '2d', '3d_v3_i1', '3d_v3_i2', '3d_v3_i3', 'ensemble']
    volumina_n_layer(data, labels)



if __name__ == '__main__':
    meta.load()

    res1 = vigra.readHDF5('/home/constantin/Work/multicut_pipeline/software/multicut_exp/rebuttal/snemi/snemi_ultimate_seglmc_myel_myelmerged.h5', 'data')
    #res2 = vigra.readHDF5('/home/constantin/Work/multicut_pipeline/software/multicut_exp/rebuttal/snemi/snemi_final_segmc_myel.h5', 'data')
    res3 = vigra.readHDF5('/home/constantin/Work/multicut_pipeline/software/multicut_exp/rebuttal/snemi/round3/snemi_final_seglmc_myel_myelmerged.h5', 'data')
    view_test(res1, res3)
