import cPickle as pickle
import os

from DataSet import DataSet

#FIXME TODO we don't need this any longer !
# class that holds all the datasets
class MetaSet(object):

    def __init__(self, meta_folder):
        if not os.path.exists(meta_folder):
            os.mkdir(meta_folder)
        self.meta_folder = meta_folder
        self.dict_file = os.path.join(meta_folder, "meta_dict.pkl")
        self.dset_dict = {}

    def save(self):
        with open(self.dict_file, 'w') as f:
            pickle.dump(self.dset_dict,f)

    def load(self):
        if os.path.exists(self.dict_file):
            with open(self.dict_file, 'r') as f:
                self.dset_dict = pickle.load(f)

    def add_dataset(self, ds):
        assert isinstance(ds, DataSet)
        self.dset_dict[str(ds)] = ds.obj_save_path

    def update_dataset(self, ds_name, ds):
        assert isinstance(ds_name, str)
        assert isinstance(ds, DataSet)
        assert ds_name in self.dset_dict.keys()
        self.dset_dict[ds_name] = ds

    def list_datasets(self):
        return self.dset_dict.keys()

    def get_dataset(self, ds_name):
        assert ds_name in self.dset_dict.keys(), ds_name + " , " + str(self.dset_dict.keys())
        return self.dset_dict[ds_name]

