import vigra
import numpy as np
from concurrent import futures


class RemoveSmallObjectsParams:

    def __init__(self,
                 size_thresh=10000,
                 relabel=True,
                 consecutive_labels=True,
                 parallelize=False,
                 max_threads=5):

        self.size_thresh = size_thresh
        self.relabel = relabel
        self.consecutive_labels = consecutive_labels
        self.parallelize = parallelize
        self.max_threads = max_threads


def pipeline_remove_small_objects(image, params=RemoveSmallObjectsParams()):

    def remove_small_objects(image, size_thresh,
                             parallelize=False, max_threads=5):

        # Get the unique values of the segmentation including counts
        uniq, counts = np.unique(image, return_counts=True)

        # Keep all uniques that have a count smaller than size_thresh
        small_objs = uniq[counts < size_thresh]
        print 'len(small_objs) == {}'.format(len(small_objs))
        large_objs = uniq[counts >= size_thresh]
        print 'len(large_objs) == {}'.format(len(large_objs))

        if parallelize:

            if len(small_objs) > len(large_objs):
                def get_mask(image, lbl):
                    return np.logical_not(image == lbl)

                with futures.ThreadPoolExecutor(max_threads) as do_stuff:
                    tasks = [do_stuff.submit(get_mask, image, x) for x in large_objs]
                mask = np.all([x.result() for x in tasks], axis=0)

            else:

                def get_mask(image, lbl):
                    return image == lbl

                with futures.ThreadPoolExecutor(max_threads) as do_stuff:
                    tasks = [do_stuff.submit(get_mask, image, x) for x in large_objs]
                mask = np.any([x.result() for x in tasks], axis=0)

            timage = np.array(image)
            print mask.shape
            timage[mask] = 0

        else:

            if len(small_objs) > len(large_objs):

                timage = np.zeros(image.shape, dtype=image.dtype)
                for lo in large_objs:
                    timage[image == lo] = lo

            else:

                timage = np.array(image)
                for so in small_objs:
                    timage[timage == so] = 0

                    # if len(small_objs) > len(large_objs):
                    #     mask = np.logical_not(np.any([image == x for x in large_objs], axis=0))
                    # else:
                    #     mask = np.any([image == x for x in small_objs], axis=0)
                    #
                    # timage = np.array(image)
                    # print mask.shape
                    # timage[mask] = 0

        return timage

    def remove_small_objects_relabel(
            image, size_thresh, relabel=True, consecutive_labels=True,
            parallelize=False, max_threads=5
    ):

        # Make sure all objects have their individual label
        if relabel:
            image = vigra.analysis.labelVolumeWithBackground(
                image.astype(np.uint32), neighborhood=6, background_value=0
            )

        # Remove objects smaller than size_thresh
        image = remove_small_objects(
            image, size_thresh, parallelize=parallelize, max_threads=max_threads
        )

        # Relabel the image for consecutive labels
        if consecutive_labels:
            vigra.analysis.relabelConsecutive(image, start_label=0, out=image)

        return image

    # Read parameters
    size_thresh = params.size_thresh
    relabel = params.relabel
    consecutive_labels = params.consecutive_labels
    parallelize = params.parallelize
    max_threads = params.max_threads

    # Remove small objects
    return remove_small_objects_relabel(
        image, size_thresh, relabel=relabel, consecutive_labels=consecutive_labels,
        parallelize=parallelize, max_threads=max_threads
    )


def pipeline_calc_feature_images(params=FeatureImageParams()):

    compute_feature_images(params)
