import numpy as np
import vigra
import itertools

# FIXME AAAAAAHHH THE HORROR
def get_faces_with_neighbors(image):

    # --- XY ---
    # w = x + 2*z, h = y + 2*z
    shpxy = (image.shape[0] + 2*image.shape[2], image.shape[1] + 2*image.shape[2])
    xy0 = (0, 0)
    xy1 = (image.shape[2],) * 2
    xy2 = (image.shape[2] + image.shape[0], image.shape[2] + image.shape[1])
    print shpxy, xy0, xy1, xy2

    # xy front face
    xyf = np.zeros(shpxy)
    xyf[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, 0]
    xyf[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[0, :, :]), 0, 1)
    xyf[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(image[-1, :, :], 0, 1)
    xyf[xy1[0]:xy2[0], 0:xy1[1]] = np.fliplr(image[:, 0, :])
    xyf[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = image[:, -1, :]

    # xy back face
    xyb = np.zeros(shpxy)
    xyb[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, -1]
    xyb[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(image[0, :, :], 0, 1)
    xyb[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[-1, :, :]), 0, 1)
    xyb[xy1[0]:xy2[0], 0:xy1[1]] = image[:, 0, :]
    xyb[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = np.fliplr(image[:, -1, :])

    # --- XZ ---
    # w = x + 2*y, h = z + 2*y
    shpxz = (image.shape[0] + 2*image.shape[1], image.shape[2] + 2*image.shape[1])
    xz0 = (0, 0)
    xz1 = (image.shape[1],) * 2
    xz2 = (image.shape[1] + image.shape[0], image.shape[1] + image.shape[2])
    print shpxz, xz0, xz1, xz2

    # xz front face
    xzf = np.zeros(shpxz)
    xzf[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, 0, :]
    xzf[0:xz1[0], xz1[1]:xz2[1]] = np.flipud(image[0, :, :])
    xzf[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = image[-1, :, :]
    xzf[xz1[0]:xz2[0], 0:xz1[1]] = np.fliplr(image[:, :, 0])
    xzf[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = image[:, :, -1]

    # xz back face
    xzb = np.zeros(shpxz)
    xzb[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, -1, :]
    xzb[0:xz1[0], xz1[1]:xz2[1]] = image[0, :, :]
    xzb[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = np.flipud(image[-1, :, :])
    xzb[xz1[0]:xz2[0], 0:xz1[1]] = image[:, :, 0]
    xzb[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = np.fliplr(image[:, :, -1])

    # --- YZ ---
    # w = y + 2*x, h = z + 2*x
    shpyz = (image.shape[1] + 2*image.shape[0], image.shape[2] + 2*image.shape[0])
    yz0 = (0, 0)
    yz1 = (image.shape[0],) * 2
    yz2 = (image.shape[0] + image.shape[1], image.shape[0] + image.shape[2])
    print shpyz, yz0, yz1, yz2

    # yz front face
    yzf = np.zeros(shpyz)
    yzf[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[0, :, :]
    yzf[0:yz1[0], yz1[1]:yz2[1]] = np.flipud(image[:, 0, :])
    yzf[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = image[:, -1, :]
    yzf[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(np.flipud(image[:, :, 0]), 0, 1)
    yzf[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(image[:, :, -1], 0, 1)

    # yz back face
    yzb = np.zeros(shpyz)
    yzb[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[-1, :, :]
    yzb[0:yz1[0], yz1[1]:yz2[1]] = image[:, 0, :]
    yzb[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = np.flipud(image[:, -1, :])
    yzb[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(image[:, :, 0], 0, 1)
    yzb[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(np.flipud(image[:, :, -1]), 0, 1)

    faces = {
        'xyf': xyf,
        'xyb': xyb,
        'xzf': xzf,
        'xzb': xzb,
        'yzf': yzf,
        'yzb': yzb
    }

    shp = image.shape
    bounds = {
        'xyf': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
        'xyb': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
        'xzf': np.s_[shp[1]:shp[1] + shp[0], shp[1]+1:shp[1] + shp[2]-1],
        'xzb': np.s_[shp[1]:shp[1] + shp[0], shp[1]+1:shp[1] + shp[2]-1],
        'yzf': np.s_[shp[0]+1:shp[0] + shp[1]-1, shp[0]+1:shp[0] + shp[2]-1],
        'yzb': np.s_[shp[0]+1:shp[0] + shp[1]-1, shp[0]+1:shp[0] + shp[2]-1]
    }

    return faces, bounds


def find_centroids(seg, dt, bounds):

    # TODO FIXME use vigra functionality instead to avoid dependency on skimage
    from skimage import morphology

    centroids = {}

    for lbl in np.unique(seg[bounds])[1:]:

        # Mask the segmentation
        mask = seg == lbl

        # Connected component analysis to detect when a label touches the border multiple times
        conncomp = vigra.analysis.labelImageWithBackground(mask.astype(np.uint32), neighborhood=8,
                                                           background_value=0)

        # Only these labels will be used for further processing
        # TODO FIXME use vigra.filters.multiBinaryOpening or vigra.filters.multiGrayscaleOpening (dunno which is appropriate here)
        opened_labels = np.unique(morphology.opening(conncomp))
        # unopened_labels = np.unique(conncomp)
        # print 'opened_labels = {}'.format(opened_labels)
        # print 'unopened_labels = {}'.format(unopened_labels)

        for l in opened_labels[1:]:

            # Get the current label object
            curobj = conncomp == l

            # Get disttancetransf of the object
            cur_dt = np.array(dt)
            cur_dt[curobj == False] = 0

            # Detect the global maximum of this object
            amax = np.amax(cur_dt)
            cur_dt[cur_dt < amax] = 0
            cur_dt[cur_dt > 0] = lbl

            # Get the coordinates of the maximum pixel(s)
            coords = np.where(cur_dt[bounds])

            # If something was found
            if coords[0].any():

                # Only one pixel is allowed to be selected
                # FIXME: This may cause a bug if two maximum pixels exist that are not adjacent (although it is very unlikely)
                coords = [int(np.mean(x)) for x in coords]

                if lbl in centroids.keys():
                    centroids[lbl].append(coords)
                else:
                    centroids[lbl] = [coords]

    return centroids


def translate_centroids_to_volume(centroids, volume_shape):
    rtrn_centers = {}

    for orientation, centers in centroids.iteritems():


        if orientation == 'xyf':
            centers = {
                lbl: [center + [0] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'xyb':
            centers = {
                lbl: [center + [volume_shape[2]-1] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'xzf':
            centers = {
                lbl: [[center[0], 0, center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'xzb':
            centers = {
                lbl: [[center[0], volume_shape[1]-1, center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'yzf':
            centers = {
                lbl: [[0, center[0], center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'yzb':
            centers = {
                lbl: [[volume_shape[0]-1, center[0], center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }

        for key, val in centers.iteritems():
            if key in rtrn_centers:
                rtrn_centers[key].extend(val)
            else:
                rtrn_centers[key] = val

    return rtrn_centers


def compute_border_contacts(
        segmentation,
        disttransf
):

    faces_seg, bounds = get_faces_with_neighbors(segmentation)
    faces_dt, _ = get_faces_with_neighbors(disttransf)

    centroids = {key: find_centroids(val, faces_dt[key], bounds[key]) for key, val in faces_seg.iteritems()}
    centroids = translate_centroids_to_volume(centroids, segmentation.shape)

    return centroids


# FIXME this seems to be very inefficient
def compute_path_end_pairs_and_labels(
        border_contacts,
        gt,
        correspondence_list
):

    # Convert border_contacts to path_end_pairs

    # TODO: Remove path end pairs under certain criteria
    # TODO: Do this only if GT is supplied
    # a) All classes: GT label pair is already in correspondence table
    # b) Class 'non-merged': Only take them for beta_0.5?
    # c) All classes: Too many pairs for one object

    pairs = []
    labels = []
    classes = []
    gt_labels = []
    for lbl, contacts in border_contacts.iteritems():

        # Get all possible combinations of path ends in one segmentation object
        ps = list(itertools.combinations(contacts, 2))

        # Pairs are found if the segmentation object has more than one path end
        if ps:

            # # For debugging take just the first item
            # ps = [ps[0]]

            # Determine the labels of both path ends
            label_pair = [sorted([gt[p[0], p[1], p[2]] for p in pair]) for pair in ps]

            # Assign a class to the paths:
            #   False if a path doesn't cross a merging site
            #   True if a path crosses a merging site
            new_classes = [bool(lp[1] - lp[0]) for lp in label_pair]

            pairs.extend(ps)
            labels.extend([lbl] * len(ps))
            classes.extend(new_classes)
            gt_labels.extend(label_pair)
            # pairs[lbl] = ps

    # Update the correspondence list
    correspondence_list.extend(gt_labels)
    correspondence_list = np.array(correspondence_list)
    # Only keep unique pairs
    b = np.ascontiguousarray(correspondence_list).view(
        np.dtype((np.void, correspondence_list.dtype.itemsize * correspondence_list.shape[1])))
    uniques = np.unique(b)
    correspondence_list = uniques.view(correspondence_list.dtype)
    correspondence_list = correspondence_list.reshape((correspondence_list.shape[0]/2, 2))

    return pairs, labels, classes, gt_labels, correspondence_list.tolist()


# FIXME this seems to be very inefficient
def compute_path_end_pairs(
        border_contacts,
):

    # Convert border_contacts to path_end_pairs
    pairs = []
    labels = []
    for lbl, contacts in border_contacts.iteritems():
        # Get all possible combinations of path ends in one segmentation object
        ps = list(itertools.combinations(contacts, 2))
        # Pairs are found if the segmentation object has more than one path end
        if ps:
            # # For debugging take just the first item
            # ps = [ps[0]]
            pairs.extend(ps)
            labels.extend([lbl] * len(ps))
    return pairs, labels


# def find_border_contacts_arr(segmentation, disttransf, tkey='bc', debug=False):
#
#     data = IPL()
#     data[tkey] = segmentation
#     data['disttransf'][tkey] = disttransf
#
#     find_border_contacts(data, (tkey,), 'rtrn', debug=debug)
#
#     return data['rtrn']
