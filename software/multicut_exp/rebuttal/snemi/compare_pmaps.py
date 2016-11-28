import vigra

from volumina_viewer import volumina_n_layer

def view_train():

    raw = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/raw/train-input.h5", "data")
    icv1 = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/probabilities/pmaps_icv1_train.h5", "data")
    ciresan = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/probabilities/pmaps_ciresan_train.h5", "data")
    gt = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/groundtruth/train-gt.h5", "data")

    volumina_n_layer([raw,icv1,ciresan,gt], ["raw","pmap-icv1","pmap-ciresan","groundtruth"])


def view_test():

    raw = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/raw/test-input.h5", "data")
    icv1 = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/probabilities/pmaps_icv1_test.h5", "data")
    ciresan = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/snemi3d_data/probabilities/pmaps_ciresan_test.h5", "data")

    volumina_n_layer([raw,icv1,ciresan], ["raw","pmap-icv1","pmap-ciresan"])


if __name__ == '__main__':
    view_train()
