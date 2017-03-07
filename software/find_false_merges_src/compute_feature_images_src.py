
import numpy as np
import vigra
import re
from concurrent import futures
import os
import h5py

class FeatureImageParams:
    def __init__(self,
                 feature_list={
                     'raw': None,
                     'gauss_2': {'params': [2], 'func': 'gaussian'},
                     'gauss_5': {'params': [5], 'func': 'gaussian'},
                     'gauss_10': {'params': [10], 'func': 'gaussian'},
                     'mag_1': {'params': [1], 'func': 'gaussian_gradient_magnitude'},
                     'mag_5': {'params': [5], 'func': 'gaussian_gradient_magnitude'},
                     'mag_10': {'params': [10], 'func': 'gaussian_gradient_magnitude'},
                     'hess_1': {'params': [1], 'func': 'hessian_eigenvalues'},
                     'hess_5': {'params': [5], 'func': 'hessian_eigenvalues'},
                     'hess_10': {'params': [10], 'func': 'hessian_eigenvalues'},
                     'lapl_1': {'params': [1], 'func': 'laplacian_of_gaussian'},
                     'lapl_5': {'params': [5], 'func': 'laplacian_of_gaussian'},
                     'lapl_10': {'params': [10], 'func': 'laplacian_of_gaussian'},
                     'structen_1_2': {'params': [1, 2], 'func': 'structure_tensor_eigenvalues'},
                     'structen_5_10': {'params': [5, 10], 'func': 'structure_tensor_eigenvalues'},
                     'structen_10_20': {'params': [10, 20], 'func': 'structure_tensor_eigenvalues'}
                 },
                 anisotropy=[1, 1, 10],
                 max_threads_features=5
                 ):
        self.feature_list=feature_list
        self.anisotropy=anisotropy
        self.max_threads_features=max_threads_features

    def get_feature_specs(self, feature_path, return_children=False):

        if feature_path == '':
            if not return_children:
                return None, None, None, True
            else:
                return None, None, None, self.feature_list.keys()

        keys = re.split("/", feature_path)

        params = self.feature_list
        for key in keys:
            params = params[key]

        parent = re.sub("[^/]+$", '', feature_path)
        parent = re.sub("/$", '', parent)
        if parent == '':
            parent = None

        if params is not None:
            function_name = params['func']
            function_params = params['params']
            has_children = False
            children = params.keys()
            if 'raw' in children:
                children.remove('raw')
            if 'params' in children:
                children.remove('params')
            if 'func' in children:
                children.remove('func')
            if len(children) > 0:
                has_children = True
        else:
            function_name = None
            function_params = None
            has_children =False

        if return_children:
            return parent, function_name, function_params, children
        else:
            return parent, function_name, function_params, has_children


# This is FeatureImageParams with a different default feature list
class SegFeatureImageParams(FeatureImageParams):
    def __init__(self,
                 feature_list={
                     'disttransf': {
                         'params': None, 'func': 'disttransf',
                         'raw': None,
                         'gauss_2': {'params': [2], 'func': 'gaussian'},
                         'gauss_5': {'params': [5], 'func': 'gaussian'},
                         'gauss_10': {'params': [10], 'func': 'gaussian'},
                         'mag_1': {'params': [1], 'func': 'gaussian_gradient_magnitude'},
                         'mag_5': {'params': [5], 'func': 'gaussian_gradient_magnitude'},
                         'mag_10': {'params': [10], 'func': 'gaussian_gradient_magnitude'},
                         'hess_1': {'params': [1], 'func': 'hessian_eigenvalues'},
                         'hess_5': {'params': [5], 'func': 'hessian_eigenvalues'},
                         'hess_10': {'params': [10], 'func': 'hessian_eigenvalues'},
                         'lapl_1': {'params': [1], 'func': 'laplacian_of_gaussian'},
                         'lapl_5': {'params': [5], 'func': 'laplacian_of_gaussian'},
                         'lapl_10': {'params': [10], 'func': 'laplacian_of_gaussian'},
                         'structen_1_2': {'params': [1, 2], 'func': 'structure_tensor_eigenvalues'},
                         'structen_5_10': {'params': [5, 10], 'func': 'structure_tensor_eigenvalues'},
                         'structen_10_20': {'params': [10, 20], 'func': 'structure_tensor_eigenvalues'}
                     }
                 },
                 anisotropy=[1, 1, 10],
                 max_threads_features=5
                 ):
        FeatureImageParams.__init__(self, feature_list, anisotropy, max_threads_features)


class FeatureFunctions:

    def __init__(self):
        pass

    @staticmethod
    def gaussian(image, sigma, anisotropy=None):

        print 'Computing gaussian ...'
        if anisotropy:
            if type(sigma) is not list and type(sigma) is not tuple and type(sigma) is not np.array:
                sigma = np.array([sigma] * 3).astype(np.float32) / anisotropy
            else:
                sigma = np.array(sigma) / anisotropy
        image = image.astype(np.float32)

        print ' ... done with gaussian'
        return vigra.filters.gaussianSmoothing(image, sigma)

    @staticmethod
    def disttransf(image, anisotropy=[1, 1, 1]):

        print 'Computing disttransf ...'

        def pixels_at_boundary(image, axes=[1, 1, 1]):
            return axes[0] * ((np.concatenate((image[(0,), :, :], image[:-1, :, :]))
                               - np.concatenate((image[1:, :, :], image[(-1,), :, :]))) != 0) \
                   + axes[1] * ((np.concatenate((image[:, (0,), :], image[:, :-1, :]), 1)
                                 - np.concatenate((image[:, 1:, :], image[:, (-1,), :]), 1)) != 0) \
                   + axes[2] * ((np.concatenate((image[:, :, (0,)], image[:, :, :-1]), 2)
                                 - np.concatenate((image[:, :, 1:], image[:, :, (-1,)]), 2)) != 0)

        anisotropy = np.array(anisotropy).astype(np.float32)
        image = image.astype(np.float32)

        # Compute boundaries
        axes = (anisotropy ** -1).astype(np.uint8)
        image = pixels_at_boundary(image, axes)

        # Compute distance transform
        image = image.astype(np.float32)
        image = vigra.filters.distanceTransform(image, pixel_pitch=anisotropy, background=True)

        print ' ... done with disttransf'

        return image

    @staticmethod
    def hessian_eigenvalues(image, scale, anisotropy=None):

        print 'Computing hessian eigenvalues ...'

        if anisotropy:
            if type(scale) is not list and type(scale) is not tuple and type(scale) is not np.array:
                scale = list(np.array([scale] * 3).astype(np.float32) / anisotropy)
            else:
                scale = list(np.array(scale) / anisotropy)

        image = image.astype(np.float32)
        result = vigra.filters.hessianOfGaussianEigenvalues(image, scale)

        print ' ... done computing hessian eigenvalues'
        return result

    @staticmethod
    def structure_tensor_eigenvalues(image, inner_scale, outer_scale, anisotropy=None):

        print 'Computing structure tensor eigenvalues ...'

        if anisotropy:
            if type(inner_scale) is not list and type(inner_scale) is not tuple and type(inner_scale) is not np.array:
                inner_scale = list(np.array([inner_scale] * 3).astype(np.float32) / anisotropy)
            else:
                inner_scale = list(np.array(inner_scale) / anisotropy)
            if type(outer_scale) is not list and type(outer_scale) is not tuple and type(outer_scale) is not np.array:
                outer_scale = list(np.array([outer_scale] * 3).astype(np.float32) / anisotropy)
            else:
                outer_scale = list(np.array(inner_scale) / anisotropy)

        image = image.astype(np.float32)

        result = vigra.filters.structureTensorEigenvalues(image, inner_scale, outer_scale)
        print ' ... done computing structure tensor eigenvalues'

        return result

    @staticmethod
    def gaussian_gradient_magnitude(image, sigma, anisotropy=None):

        print 'Computing gradient magnitude ...'

        if anisotropy:
            if type(sigma) is not list and type(sigma) is not tuple and type(sigma) is not np.array:
                sigma = np.array([sigma] * 3).astype(np.float32) / anisotropy
            else:
                sigma = np.array(sigma) / anisotropy

        image = image.astype(np.float32)
        result = vigra.filters.gaussianGradientMagnitude(image, sigma)
        print ' ... done computing gradient magnitude'
        return result

    @staticmethod
    def laplacian_of_gaussian(image, sigma, anisotropy=None):

        print 'Computing Laplacian ...'

        if anisotropy:
            if type(sigma) is not list and type(sigma) is not tuple and type(sigma) is not np.array:
                sigma = list(np.array([sigma] * 3).astype(np.float32) / anisotropy)
            else:
                sigma = list(np.array(sigma) / anisotropy)

        image = image.astype(np.float32)
        result = vigra.filters.laplacianOfGaussian(image, sigma)
        print ' ... done computing Laplacian'
        return result


class FeatureImages(FeatureFunctions):

    def __init__(self, source_filepath=None, source_internal_path=None,
                 filepath=None, internal_path='', params=FeatureImageParams()):

        FeatureFunctions.__init__(self)

        self._source_filepath = source_filepath
        self._source_internal_path = source_internal_path

        if filepath is not None:
            self._filepath = filepath
        else:
            self._filepath = None

        self._internal_path = internal_path
        self._params = params

        self._f = None

    def compute_feature(self, path_to_feature, return_nothing=False):

        parent, function, params, has_children = self._params.get_feature_specs(path_to_feature)

        if parent is not None:
            parent_image = self.get_feature(parent)
        else:
            # parent_image = vigra.impex.readHDF5(self._source_filepath, self._source_internal_path)
            with h5py.File(self._source_filepath, mode='r') as f:
                parent_image = np.array(f[self._source_internal_path])

        if function is not None:
            if params is None:
                params = []
            feature_image = getattr(self, function)(parent_image, *params, anisotropy=self._params.anisotropy)
            if has_children:
                print 'Writing to: \n    {}'.format(self._internal_path + path_to_feature + '/raw')

                if self._f is None:
                    with h5py.File(self._filepath, mode='w') as f:
                        f.create_dataset(self._internal_path + path_to_feature + '/raw', data=feature_image)
                else:
                    self._f.create_dataset(self._internal_path + path_to_feature + '/raw', data=feature_image)
                # vigra.impex.writeHDF5(feature_image, self._filepath, self._internal_path + path_to_feature + '/raw')
            else:
                print 'Writing to: \n    {}'.format(self._internal_path + path_to_feature)
                # with h5py.File(self._filepath, mode='w') as f:
                if self._f is None:
                    with h5py.File(self._filepath, mode='w') as f:
                        f.create_dataset(self._internal_path + path_to_feature, data=feature_image)
                else:
                    self._f.create_dataset(self._internal_path + path_to_feature, data=feature_image)
                # vigra.impex.writeHDF5(feature_image, self._filepath, self._internal_path + path_to_feature)

            if not return_nothing:
                return feature_image

        else:
            if not return_nothing:
                return parent_image

    def get_feature(self, path_to_feature, return_nothing=False):

        filepath = self._filepath
        # TODO os.path.exists(filepath)

        if not os.path.exists(filepath):

            return self.compute_feature(path_to_feature, return_nothing=return_nothing)

        else:

            # with h5py.File(filepath, mode='r') as f:
            #     if self._internal_path + path_to_feature + '/raw' in f and type(f[self._internal_path + path_to_feature + '/raw']) is h5py.Dataset:
            #         # print '{} found in f'.format(self._internal_path + path_to_feature + '/raw')
            #         if not return_nothing:
            #             return vigra.readHDF5(filepath, self._internal_path + path_to_feature + '/raw')
            #     elif self._internal_path + path_to_feature in f and type(f[self._internal_path + path_to_feature]) is h5py.Dataset:
            #         # print '{} found in f'.format(self._internal_path + path_to_feature)
            #         if not return_nothing:
            #             return vigra.readHDF5(filepath, self._internal_path + path_to_feature)
            #     else:
            #         # print '{} not found in f'.format(path_to_feature)
            #         return self.compute_feature(path_to_feature, return_nothing=return_nothing)

            def check_file_content(f):

                if self._internal_path + path_to_feature + '/raw' in f and type(
                        f[self._internal_path + path_to_feature + '/raw']) is h5py.Dataset:
                    # print '{} found in f'.format(self._internal_path + path_to_feature + '/raw')
                    if not return_nothing:
                        # return vigra.readHDF5(filepath, self._internal_path + path_to_feature + '/raw')
                        return np.array(f[self._internal_path + path_to_feature + '/raw'])
                elif self._internal_path + path_to_feature in f and type(
                        f[self._internal_path + path_to_feature]) is h5py.Dataset:
                    # print '{} found in f'.format(self._internal_path + path_to_feature)
                    if not return_nothing:
                        # return vigra.readHDF5(filepath, self._internal_path + path_to_feature)
                        return np.array(f[self._internal_path + path_to_feature])
                else:
                    # print '{} not found in f'.format(path_to_feature)
                    return self.compute_feature(path_to_feature, return_nothing=return_nothing)

            if self._f is None:
                with h5py.File(filepath, mode='r') as f:
                    return check_file_content(f)
            else:
                return check_file_content(self._f)

            # what_to_do = 'nothing'
            # with h5py.File(filepath, mode='r') as f:
            #     if self._internal_path + path_to_feature + '/raw' in f and type(
            #             f[self._internal_path + path_to_feature + '/raw']) is h5py.Dataset:
            #         what_to_do = 'read_with_raw'
            #     elif self._internal_path + path_to_feature in f and type(
            #             f[self._internal_path + path_to_feature]) is h5py.Dataset:
            #         what_to_do = 'read'
            #     else:
            #         what_to_do = 'compute'
            #
            # if what_to_do == 'read':
            #     if not return_nothing:
            #         return vigra.impex.readHDF5(filepath, self._internal_path + path_to_feature)
            # elif what_to_do == 'read_with_raw':
            #     if not return_nothing:
            #         return vigra.impex.readHDF5(filepath, self._internal_path + path_to_feature + '/raw')
            # elif what_to_do == 'compute':
            #     return self.compute_feature(path_to_feature, return_nothing=return_nothing)

        # # file.keys()
        # try:
        #     # Let's see if the feature is already cached
        #     return vigra.readHDF5(filepath, self._internal_path + path_to_feature)
        # except (KeyError, IOError):
        #     try:
        #         # Maybe it was a feature with children, then it is stored like this
        #         return vigra.readHDF5(filepath, self._internal_path + path_to_feature + '/raw')
        #     except (KeyError, IOError):
        #         # Or maybe it was not yet computed
        #         return self.compute_feature(path_to_feature)

    def compute_children(self, path_to_parent='', parallelize=True):
        """
        Caches all child features of the specified parent
        To cache all feature images set path_to_parent='' (default)
        :param path_to_parent:
        :param parallelize:
        :return:
        """

        import time

        if self._params.max_threads_features == 1:
            parallelize = False

        # This is used to organize the threads
        self._available_threads = self._params.max_threads_features

        def parallelizing_wrapper(path_to_feature):

            # Get the children of the current parent
            parent, function, params, children = self._params.get_feature_specs(path_to_feature, return_children=True)

            # This ensures that the respective feature is computed if it is not already chached
            self.get_feature(path_to_feature, return_nothing=True)
            # Once the calculation and caching is done the thread can be made available for other tasks
            self._available_threads += 1

            if not children:
                return

            if not parallelize:
                # Non-parallelized version
                for child in children:

                    if path_to_feature != '':
                        child = path_to_feature + '/' + child

                    parallelizing_wrapper(child)

            else:

                while children:

                    # Parallelized version
                    # --------------------

                    # Check how many children still need to be computed and how many threads are still
                    #   available
                    # Only open as many threads as necessary and submit only as many children as threads
                    #   are available
                    if len(children) > self._available_threads:
                        start_threads = self._available_threads
                        working_children = children[:start_threads]
                        children = children[start_threads:]
                    else:
                        start_threads = len(children)
                        working_children = children
                        children = []
                    # Block the threads by reducing the remaining available thread count
                    self._available_threads -= start_threads
                    print 'available_threads = {}'.format(self._available_threads)

                    if start_threads > 0:
                        with futures.ThreadPoolExecutor(start_threads) as do_stuff:
                            tasks = []
                            for child in working_children:
                                if path_to_feature != '':
                                    child = path_to_feature + '/' + child
                                tasks.append(
                                    do_stuff.submit(
                                        parallelizing_wrapper, child
                                    )
                                )
                        print 'available_threads = {}'.format(self._available_threads)
                    else:
                        print 'No threads available: Sleeping for 1 second and checking again'
                        time.sleep(1)

        self._f = h5py.File(self._filepath, mode='a')

        self._available_threads -= 1
        # tasks = create_task_list(path_to_feature)
        parallelizing_wrapper(path_to_parent)

        self._f.close()
        self._f = None



