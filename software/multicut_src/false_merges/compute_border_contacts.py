from __future__ import print_function, division

import numpy as np
import vigra
import itertools

from ..Postprocessing import remove_small_segments

#######################
# new border extraction
#######################


# for face and line numberings in cube, see:
# https://drive.google.com/file/d/0B4_sYa95eLJ1VFZ5VjJtQlhXcEE/view?usp=sharing
class Cube(object):

    def __init__(self, shape):
        assert len(shape) == 3
        self.shape = shape
        self.n_faces = 6
        self.n_lines = 12
        # list of dicts matching lines belonging to relative positions in faces
        # 0 -> front in first dim, 1 -> front in second dim, 3 -> front in second dim, 4 -> back in second dim
        self._lines_to_faces = [
            {0: 0, 1: 3,  2: 2,  3: 1},  # face 0 -> lower z face
            {0: 0, 4: 1,  7: 3,  8: 2},  # face 1 -> front y face
            {1: 0, 6: 1,  7: 3,  9: 2},  # face 2 -> right x face
            {2: 0, 5: 1,  6: 3, 10: 2},  # face 3 -> back y face
            {3: 0, 4: 3,  5: 1, 11: 2},  # face 4 -> left x face
            {8: 0, 9: 3, 10: 2, 11: 1}   # face 5 -> upper z face,
        ]
        self._faces_to_lines = [
            [0, 1],  # line 0
            [0, 2],  # line 1
            [0, 3],  # line 2
            [0, 4],  # line 3
            [1, 4],  # line 4
            [3, 4],  # line 5
            [2, 3],  # line 6
            [1, 2],  # line 7
            [1, 5],  # line 8
            [2, 5],  # line 9
            [3, 5],  # line 10
            [4, 5]   # line 11
        ]

    def slice_from_line_id(self, line_id):
        assert line_id < self.n_lines
        # dim -> the dimension in which the coordinates varies
        dim = 0 if line_id in (0, 2, 8, 10) else (1 if line_id in (1, 3, 9, 11) else 2)
        # select 1 -> determines whether the first fixed coordinate is at the origin or at the end
        select1 = 0 if line_id in (0, 3, 4, 5, 8, 11) else -1
        # select 2 -> determines whether the second fixed coordinate is at the origin or at the end
        select2 = 0 if line_id in (0, 1, 2, 3, 4, 7) else -1
        if dim == 0:
            return np.s_[:, select1, select2]
        elif dim == 1:
            return np.s_[select1, :, select2]
        elif dim == 2:
            return np.s_[select1, select2, :]

    def slice_from_face_id(self, face_id):
        assert face_id < self.n_faces
        # dim -> the dimension in which the face is fixed
        dim = 0 if face_id in (2, 4) else (1 if face_id in (1, 3) else 2)
        # select -> determines whether dim is fixed at origin or at the end
        select = 0 if face_id in (0, 1, 4) else -1
        if dim == 0:
            return np.s_[select, :, :]
        elif dim == 1:
            return np.s_[:, select, :]
        elif dim == 2:
            return np.s_[:, :, select]

    def line_ids_from_face_id(self, face_id):
        assert face_id < self.n_faces
        return self._lines_to_faces[face_id].keys()

    def face_ids_from_line_id(self, line_id):
        assert line_id < self.n_lines
        return self._faces_to_lines[line_id]

    def line_slice_from_face_id(self, face_id, line_id):
        assert face_id < self.n_faces
        assert line_id in self._lines_to_faces[face_id].keys()
        i = self._lines_to_faces[face_id][line_id]
        # select -> determines whether line is at front or back of slice
        select = 0 if i > 1 else -1
        if i % 2 == 0:
            return np.s_[:, select]
        else:
            return np.s_[select, :]

    def centroids_face_to_vol(self, centroids, face_id):
        assert face_id < self.n_faces
        # dim -> the dimension in which the face is fixed
        dim = 0 if face_id in (2, 4) else (1 if face_id in (1, 3) else 2)
        # select -> determines whether dim is fixed at origin or at the end
        extra_coord_val = 0 if face_id in (0, 1, 4) else self.shape[dim] - 1
        if dim == 0:
            return [(extra_coord_val,) + centr for centr in centroids]
        elif dim == 1:
            return [(centr[0], extra_coord_val, centr[1]) for centr in centroids]
        elif dim == 2:
            return [centr + (extra_coord_val,) for centr in centroids]


# FIXME merge along lines == True not tested yet
def compute_border_contacts(seg, merge_along_lines=False):

    cube = Cube(seg.shape)
    min_size = 4 * 4  # removing smaller then 4*4 pixel segmentes -> TODO maybe this should be exposed ?!

    centroid_offset = 0
    centroid_list  = []
    centroid_sizes = []
    centroid_ids_to_labels = {}

    centroid_id_lines = []

    for face_id in range(cube.n_faces):

        # get the slice for this face and extract the corresponding labels
        face = cube.slice_from_face_id(face_id)
        # find the centroids for all unique labels in this face
        seg_face = seg[face]

        # relabel and remove small border contacts
        seg_labeled, seg_sizes = remove_small_segments(
            seg_face,
            size_thresh=min_size,
            relabel=True,
            return_sizes=True
        )

        # get the centroids via vigra eccentricity centers
        centroids   = vigra.filters.eccentricityCenters(seg_labeled)[1:]

        # associate centroid ids with labels
        for i, centr in enumerate(centroids):
            centroid_ids_to_labels[i + centroid_offset] = seg_face[centr]

        # if we merge the centroids along the lines of the cube later,
        # we now assign each line pixel to its corresponding centroid id
        if merge_along_lines:

            # match the relabeled segments to centroids
            other_labels_to_centroid_ids = {
                seg_labeled[centr]: i + centroid_offset for i, centr in enumerate(centroids)
            }
            line_id_to_centroid_lines = {}

            # for each line adjacent to this face, make an array that records the centroid id for each
            # pixel on the line
            for line_id in cube.line_ids_from_face_id(face_id):
                line = cube.line_slice_from_face_id(face_id, line_id)
                seg_line = seg_labeled[line]
                line_to_centroid_ids = np.zeros_like(seg_line, dtype='int32')
                for ii, label in enumerate(seg_line):
                    line_to_centroid_ids[ii] = other_labels_to_centroid_ids[label] if label != 0 else -1
                line_id_to_centroid_lines[line_id] = line_to_centroid_ids

            centroid_id_lines.append(line_id_to_centroid_lines)
            centroid_sizes.extend([seg_sizes[seg_labeled[centr]] for centr in centroids])

        # extend centroid list with centroids mapped to global coordinates
        centroid_list.extend(cube.centroids_face_to_vol(centroids, face_id))
        centroid_offset += len(centroids)

    # FIXME this neglects some edge cases:
    # - large segments that are in more than 2 lines

    # if merge_along_lines is True, we iterate over the 12 lines of the cube and
    # merge the centroids of the adjacent labels, to avoid duplicate centroids of 'bend-over' segments
    ignore_centroid_ids = []
    if merge_along_lines:

        # for each line, get the adjacent faces and the 2 lines with centroid ids
        # discard the smaller of the 2 centroids sharing a line pixel if the labels agree
        for line_id in range(cube.n_lines):
            face_id1, face_id2 = cube.face_ids_from_line_id(line_id)
            line1 = centroid_id_lines[face_id1][line_id]
            line2 = centroid_id_lines[face_id2][line_id]
            assert len(line1) == len(line2)

            visited = []
            # iterate over the line and discard one of the centroids of 2 centroids with same id
            for ii, cent_id1 in enumerate(line1):
                cent_id2 = line2[ii]
                if not (cent_id1, cent_id2) in visited and cent_id1 != -1 and cent_id2 != -1:

                    if centroid_ids_to_labels[cent_id1] == centroid_ids_to_labels[cent_id2]:
                        size1, size2 = centroid_sizes[cent_id1], centroid_sizes[cent_id2]
                        ignore_centroid_ids.append(cent_id1 if size1 < size2 else cent_id2)

                    visited.append((cent_id1, cent_id2))

    # now we invert the centroids to labels, potentially leaving out ignore ids
    labels_to_centroids = {}
    for centroid_id, label in centroid_ids_to_labels.items():
        if centroid_id not in ignore_centroid_ids:
            labels_to_centroids.setdefault(label, []).append(centroid_list[centroid_id])

    # if we still have a zero-label, remove it
    if 0 in labels_to_centroids:
        del labels_to_centroids[0]

    # remove paths with only 1 end point
    for label in labels_to_centroids:
        if len(labels_to_centroids[label]) == 1:
            del labels_to_centroids[label]

    return labels_to_centroids


#######################
# old border extraction
#######################


# FIXME AAAAAAHHH THE HORROR
def get_faces_with_neighbors(image):

    # --- XY ---
    # w = x + 2*z, h = y + 2*z
    shpxy = (image.shape[0] + 2 * image.shape[2], image.shape[1] + 2 * image.shape[2])
    xy0 = (0, 0)
    xy1 = (image.shape[2],) * 2
    xy2 = (image.shape[2] + image.shape[0], image.shape[2] + image.shape[1])
    print(shpxy, xy0, xy1, xy2)

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
    shpxz = (image.shape[0] + 2 * image.shape[1], image.shape[2] + 2 * image.shape[1])
    xz0 = (0, 0)
    xz1 = (image.shape[1],) * 2
    xz2 = (image.shape[1] + image.shape[0], image.shape[1] + image.shape[2])
    print(shpxz, xz0, xz1, xz2)

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
    shpyz = (image.shape[1] + 2 * image.shape[0], image.shape[2] + 2 * image.shape[0])
    yz0 = (0, 0)
    yz1 = (image.shape[0],) * 2
    yz2 = (image.shape[0] + image.shape[1], image.shape[0] + image.shape[2])
    print(shpyz, yz0, yz1, yz2)

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
        'xzf': np.s_[shp[1]:shp[1] + shp[0], shp[1] + 1:shp[1] + shp[2] - 1],
        'xzb': np.s_[shp[1]:shp[1] + shp[0], shp[1] + 1:shp[1] + shp[2] - 1],
        'yzf': np.s_[shp[0] + 1:shp[0] + shp[1] - 1, shp[0] + 1:shp[0] + shp[2] - 1],
        'yzb': np.s_[shp[0] + 1:shp[0] + shp[1] - 1, shp[0] + 1:shp[0] + shp[2] - 1]
    }

    return faces, bounds


def find_centroids(seg, dt, bounds):

    centroids = {}

    for lbl in np.unique(seg[bounds])[1:]:

        # Mask the segmentation
        mask = seg == lbl

        # Connected component analysis to detect when a label touches the border multiple times
        conncomp = vigra.analysis.labelImageWithBackground(
            mask.astype(np.uint32),
            neighborhood=8,
            background_value=0
        )

        # Only these labels will be used for further processing
        # FIXME expose radius as parameter
        opened_labels = np.unique(vigra.filters.discOpening(conncomp.astype(np.uint8), 2))

        # unopened_labels = np.unique(conncomp)
        # print 'opened_labels = {}'.format(opened_labels)
        # print 'unopened_labels = {}'.format(unopened_labels)

        # FIXME why do we use the dt here ?
        # as far as I can see, this only takes the mean of each coordinate anyway
        # -> probably best to take the eccentricity centers instead
        for l in opened_labels[1:]:

            # Get the current label object
            curobj = conncomp == l

            # Get disttancetransf of the object
            cur_dt = dt.copy()
            cur_dt[curobj==False] = 0

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

    for orientation, centers in centroids.items():

        if orientation == 'xyf':
            centers = {
                lbl: [center + [0] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.items()
            }
        elif orientation == 'xyb':
            centers = {
                lbl: [center + [volume_shape[2] - 1] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.items()
            }
        elif orientation == 'xzf':
            centers = {
                lbl: [[center[0], 0, center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.items()
            }
        elif orientation == 'xzb':
            centers = {
                lbl: [[center[0], volume_shape[1] - 1, center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.items()
            }
        elif orientation == 'yzf':
            centers = {
                lbl: [[0, center[0], center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.items()
            }
        elif orientation == 'yzb':
            centers = {
                lbl: [[volume_shape[0] - 1, center[0], center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.items()
            }

        for key, val in centers.items():
            if key in rtrn_centers:
                rtrn_centers[key].extend(val)
            else:
                rtrn_centers[key] = val

    return rtrn_centers


def compute_border_contacts_old(
        segmentation,
        disttransf
):

    faces_seg, bounds = get_faces_with_neighbors(segmentation)
    faces_dt, _ = get_faces_with_neighbors(disttransf)

    centroids = {key: find_centroids(val, faces_dt[key], bounds[key]) for key, val in faces_seg.items()}
    centroids = translate_centroids_to_volume(centroids, segmentation.shape)

    return centroids


####################################
# extract pairs from border contacts
####################################


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

    use_correspondence = False
    if correspondence_list is not None:
        use_correspondence = True

    pairs = []
    labels = []
    classes = []
    gt_labels = []
    for lbl, contacts in border_contacts.items():

        # Get all possible combinations of path ends in one segmentation object
        ps = list(itertools.combinations(contacts, 2))

        # Pairs are found if the segmentation object has more than one path end
        if ps:

            # # For debugging take just the first item
            # ps = [ps[0]]

            # Determine the labels of both path ends
            label_pair = [sorted([gt[p[0], p[1], p[2]] for p in pair]) for pair in ps]

            # Throw out pairs if they were already found in a different mc source
            if use_correspondence:
                corr_mask = np.array([x not in correspondence_list for x in label_pair])
                label_pair = np.array(label_pair)[corr_mask, ...].tolist()
                ps = np.array(ps)[corr_mask, ...].tolist()

            # The path list can, again, be empty
            if ps:
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
    correspondence_list = correspondence_list.reshape((correspondence_list.shape[0] // 2, 2))

    return np.array(pairs), np.array(labels), np.array(classes), np.array(gt_labels), correspondence_list.tolist()


def compute_path_end_pairs(border_contacts):

    # Convert border_contacts to path_end_pairs
    pairs = []
    labels = []
    for lbl, contacts in border_contacts.items():
        # Get all possible combinations of path ends in one segmentation object
        ps = list(itertools.combinations(contacts, 2))
        # Pairs are found if the segmentation object has more than one path end
        if ps:
            # # For debugging take just the first item
            # ps = [ps[0]]
            pairs.extend(ps)
            labels.extend([lbl] * len(ps))
    return np.array(pairs), np.array(labels)
