import numpy as np
import vigra

from wsDtSegmentation import wsDtSegmentation


# make 2d superpixel based on watershed on DT
def ws_distrafo_debug(probs, thresh, sig_seeds, sig_weights, clean_seeds = False):

    segmentation = np.zeros_like(probs, dtype = np.uint32)
    mask = np.zeros_like(probs, dtype = np.uint32)
    binary_seeds = np.zeros_like(probs, dtype = np.uint32)
    labeled_seeds = np.zeros_like(probs, dtype = np.uint32)
    dt_seeds = np.zeros_like(probs)
    dt = np.zeros_like(probs)
    offset = 0
    for z in xrange(probs.shape[2]):
        out_dict = {}
        wsdt = wsDtSegmentation(probs[:,:,z], thresh,
                25, 5,
                sig_seeds, sig_weights,
                clean_seeds, out_dict)
        mask[:,:,z] = out_dict['thresholded membranes'].checkoutSubarray( (0,0), (512,512) ).astype(np.uint32)
        binary_seeds[:,:,z] = out_dict['binary seeds'].checkoutSubarray( (0,0), (512,512) ).astype(np.uint32)

        labeled_seeds[:,:,z] = out_dict['seeds'].checkoutSubarray( (0,0), (512,512) ).astype(np.uint32)
        dt[:,:,z] = out_dict['distance transform'].checkoutSubarray( (0,0), (512,512) )
        if 'smoothed DT for seeds' in out_dict:
            dt_seeds[:,:,z] = out_dict['smoothed DT for seeds'].checkoutSubarray( (0,0), (512,512) )

        segmentation[:,:,z] = wsdt
        segmentation[:,:,z] += offset
        offset = np.max(segmentation)
    return segmentation, mask, dt, dt_seeds, binary_seeds, labeled_seeds

# make 2d superpixel based on watershed on DT
def ws_distrafo(probs, thresh, sig_seeds, sig_weights, clean_seeds = True):

    segmentation = np.zeros_like(probs, dtype = np.uint32)
    offset = 0
    for z in xrange(probs.shape[2]):
        out_dict = {}
        wsdt = wsDtSegmentation(probs[:,:,z], thresh,
                1, 1,
                sig_seeds, sig_weights,
                clean_seeds)
        segmentation[:,:,z] = wsdt
        segmentation[:,:,z] += offset
        offset = np.max(segmentation)
    return segmentation


# watershed on smoothed probability maps
def ws_smoothed(probs, sigma):

    segmentation = np.zeros_like(probs, dtype = np.uint32)
    offset = 0
    for z in xrange(probs.shape[2]):
        hmap = vigra.filters.gaussianSmoothing(probs[:,:,z], sigma)

        seeds = vigra.analysis.localMinima( hmap.astype(np.float32),
                neighborhood = 8,
                allowPlateaus=True,
                allowAtBorder=True).astype(np.uint32)

        seeds = vigra.analysis.labelImageWithBackground(seeds)

        segmentation[:,:,z] = vigra.analysis.watershedsNew(hmap,
           seeds = seeds,
           neighborhood = 8)[0]
        segmentation[:,:,z] += offset
        offset = np.max(segmentation[:,:,z])

    return segmentation


def eval_superpix(seg):
    import vigra.graphs as vgraph
    rag = vgraph.regionAdjacencyGraph( vgraph.gridGraph(seg.shape[0:3]), seg )
    print "Number Nodes:", rag.nodeNum
    print "Number Edges:", rag.edgeNum



if __name__ == '__main__':

    prob_path_train = "/home/consti/Work/nature_experiments/isbi12_data/probabilities/interceptor_train.h5"
    prob_path_test = "/home/consti/Work/nature_experiments/isbi12_data/probabilities/interceptor_test.h5"

    prob_key = "data"

    #params for watersheds on distancetrafo
    p_thresh = 0.3
    sig_seeds = 2.0
    sig_weights = 2.6

    # params for vanilla supepixel
    sigma = 2.0

    probs = vigra.readHDF5(prob_path_test, prob_key)

    #seg_wsdt, mask, dt, dt_seeds, binary_seeds, labeled_seeds = ws_distrafo_debug(probs, p_thresh, sig_seeds, sig_weights, False)
    seg_wsdt = ws_distrafo(probs, p_thresh, sig_seeds, sig_weights, False)
    seg_ws   = ws_smoothed(probs, sigma)

    print "wsdt"
    eval_superpix(seg_wsdt)
    print "ws"
    eval_superpix(seg_ws)

    vigra.writeHDF5(seg_wsdt,
            "/home/consti/Work/nature_experiments/isbi12_data/watersheds/wsdt_interceptor_test.h5",
            "data")
    vigra.writeHDF5(seg_ws,
            "/home/consti/Work/nature_experiments/isbi12_data/watersheds/wssmoothed_interceptor_test.h5",
            "data")

    #from volumina_viewer import volumina_n_layer
    #raw = vigra.readHDF5("/home/consti/Work/nature_experiments/isbi12_data/raw/test-volume.h5",
    #        "data")
    #volumina_n_layer([raw,probs, seg_wsdt, seg_ws])
    #volumina_n_layer([probs, mask, dt,  dt_seeds, binary_seeds, seg_wsdt.astype(np.uint32)] )
