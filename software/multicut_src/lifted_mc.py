import os

import numpy
import vigra

import vigra.graphs as vgraph
import graph as agraph

from MCSolverImpl import *
from Tools import cacher_hdf5
from EdgeRF import learn_and_predict_rf_from_gt

from defect_handling import modified_mc_problem

RandomForest = vigra.learning.RandomForest3


@cacher_hdf5(ignoreNumpyArrays=True)
def clusteringFeatures(ds,
        segId,
        extraUV,
        edgeIndicator,
        liftedNeighborhood,
        is_perturb_and_map = False,
        with_defects = False):

    print "Computing clustering features for lifted neighborhood", liftedNeighborhood
    if is_perturb_and_map:
        print "For perturb and map"
    else:
        print "For normal clustering"

    if with_defects:
        n_nodes, uvs_local = modified_mc_problem(ds, segId)
        originalGraph = vgraph.listGraph(n_nodes)
        originalGraph.addEdges(uvs_local)
    else:
        originalGraph = ds._rag(segId)
    extraUV = numpy.require(extraUV,dtype='uint32')
    uvOriginal = originalGraph.uvIds()
    liftedGraph =  vgraph.listGraph(originalGraph.nodeNum)
    liftedGraph.addEdges(uvOriginal)
    liftedGraph.addEdges(extraUV)

    uvLifted = liftedGraph.uvIds()
    foundEdges = originalGraph.findEdges(uvLifted)
    foundEdges[foundEdges>=0] = 0
    foundEdges *= -1

    nAdditionalEdges = liftedGraph.edgeNum - originalGraph.edgeNum
    whereLifted = numpy.where(foundEdges==1)[0].astype('uint32')
    assert len(whereLifted) == nAdditionalEdges
    assert foundEdges.sum() == nAdditionalEdges

    allFeat = []
    eLen = vgraph.getEdgeLengths(originalGraph)
    nodeSizes_ = vgraph.getNodeSizes(originalGraph)
    #for wardness in [0.01, 0.6, 0.7]:
    for wardness in [0.01, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7]:

        edgeLengthsNew = numpy.concatenate([eLen,numpy.zeros(nAdditionalEdges)]).astype('float32')
        edgeIndicatorNew = numpy.concatenate([edgeIndicator,numpy.zeros(nAdditionalEdges)]).astype('float32')

        nodeSizes = nodeSizes_.copy()
        nodeLabels = vgraph.graphMap(originalGraph,'node',dtype='uint32')

        nodeFeatures = vgraph.graphMap(liftedGraph,'node',addChannelDim=True)
        nodeFeatures[:]=0


        outWeight=vgraph.graphMap(liftedGraph,item='edge',dtype=numpy.float32)


        mg = vgraph.mergeGraph(liftedGraph)
        clusterOp = vgraph.minEdgeWeightNodeDist(mg,
            edgeWeights=edgeIndicatorNew,
            edgeLengths=edgeLengthsNew,
            nodeFeatures=nodeFeatures,
            nodeSizes=nodeSizes,
            nodeLabels=nodeLabels,
            beta=0.5,
            metric='l1',
            wardness=wardness,
            outWeight=outWeight
        )


        clusterOp.setLiftedEdges(whereLifted)


        hc = vgraph.hierarchicalClustering(clusterOp,nodeNumStopCond=1,
                                            buildMergeTreeEncoding=False)
        hc.cluster()

        assert mg.nodeNum == 1
        assert mg.edgeNum == 0
        tweight = edgeIndicatorNew.copy()
        hc.ucmTransform(tweight)

        whereInLifted = liftedGraph.findEdges(extraUV)
        assert whereInLifted.min() >= 0
        feat = tweight[whereInLifted]
        assert feat.shape[0] == extraUV.shape[0]
        allFeat.append(feat[:,None])

    weights = numpy.concatenate(allFeat,axis=1)
    mean = numpy.mean(weights,axis=1)[:,None]
    stddev = numpy.std(weights,axis=1)[:,None]
    allFeat = numpy.concatenate([weights,mean,stddev],axis=1)

    return numpy.nan_to_num(allFeat)

#
# Multicut features
#

# TODO we might even loop over different solver here ?! -> nifty greedy ?!
@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_multiple_segmentations(ds,
        trainsets,
        referenceSegId,
        featureListLocal,
        uvIdsLifted,
        pipelineParam):

    assert False, "Currently not supported"
    import nifty
    from concurrent import futures

    print "Computing lifted features from multople segmentations from %i segmentations for reference segmentation %i" % (ds.n_seg, referenceSegId)

    pLocals = []
    indications = []
    rags = []
    for candidateSegId in xrange(ds.n_seg):
        pLocals.append( learn_and_predict_rf_from_gt(pipelineParam.rf_cache_folder,
            trainsets, ds,
            candidateSegId, candidateSegId,
            featureListLocal, pipelineParam) )
        # append edge indications (strange hdf5 bugs..)
        rags.append(ds._rag(candidateSegId))
        indications.append(ds.edge_indications(candidateSegId))


    def single_mc(segId, pLocal, edge_indications, rag, beta):

        weight = pipelineParam.weight

        # copy the probabilities
        probs = pLocal.copy()
        p_min = 0.001
        p_max = 1. - p_min
        probs = (p_max - p_min) * probs + p_min
        # probs to energies
        energies = np.log( (1. - probs) / probs ) + np.log( (1. - beta) / beta )

        edge_areas = rag.edgeLengths()

        # weight the energies
        if pipelineParam.weighting_scheme == "z":
            print "Weighting Z edges"
            # z - edges are indicated with 0 !
            area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
            # we only weight the z edges !
            w = weight * edge_areas[edge_indications == 0] / area_z_max
            energies[edge_indications == 0] = np.multiply(w, energies[edge_indications == 0])

        elif pipelineParam.weighting_scheme == "xyz":
            print "Weighting xyz edges"
            # z - edges are indicated with 0 !
            area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
            len_xy_max = float( np.max( edge_areas[edge_indications == 1] ) )
            # weight the z edges !
            w_z = weight * edge_areas[edge_indications == 0] / area_z_max
            energies[edge_indications == 0] = np.multiply(w_z, energies[edge_indications == 0])
            # weight xy edges
            w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
            energies[edge_indications == 1] = np.multiply(w_xy, energies[edge_indications == 1])

        elif pipelineParam.weighting_scheme == "all":
            print "Weighting all edges"
            area_max = float( np.max( edge_areas ) )
            w = weight * edge_areas / area_max
            energies = np.multiply(w, energies)

        uv_ids = np.sort(rag.uvIds(), axis = 1)
        n_var = uv_ids.max() + 1

        mc_node, mc_energy, t_inf = multicut_fusionmoves(
                n_var, uv_ids,
                energies, pipelineParam)

        return mc_node


    def map_nodes_to_reference(segId, nodes, rag, ragRef):

        if segId == referenceSegId:
            return nodes

        # map nodes to segmentation
        denseSegmentation = rag.projectLabelsToBaseGraph(nodes.astype('uint32'))
        # map segmentation to nodes of reference seg
        referenceNodes, _ = ragRef.projectBaseGraphGt(denseSegmentation)

        return referenceNodes

    #workers = 1
    workers = pipelineParam.n_threads

    # iterate over several betas
    betas = [.35,.4,.45,.5,.55,.6,.65]

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = []
        for candidateSegId in xrange(ds.n_seg):
            for beta in betas:
                tasks.append( executor.submit( single_mc, candidateSegId, pLocals[candidateSegId], indications[candidateSegId], rags[candidateSegId], beta) )

    mc_nodes = [future.result() for future in tasks]

    # map nodes to the reference seg
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = []
        for candidateSegId in xrange(ds.n_seg):
            for i in xrange(len(betas)):
                resIndex = candidateSegId * len(betas) + i
                tasks.append( executor.submit( map_nodes_to_reference, candidateSegId, mc_nodes[resIndex], rags[candidateSegId], rags[referenceSegId] ) )

    reference_nodes = [future.result() for future in tasks]

    # map multicut result to lifted edges
    allFeat = [ ( reference_node[uvIdsLifted[:, 0]] !=  reference_node[uvIdsLifted[:, 1]] )[:,None] for reference_node in reference_nodes]

    mcStates = numpy.concatenate(allFeat, axis=1)
    stateSum = numpy.sum(mcStates,axis=1)
    return numpy.concatenate([mcStates,stateSum[:,None]],axis=1)


# TODO need to adapt to defects
@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_multicut(ds,
        segId,
        pLocal,
        pipelineParam,
        uvIds,
        liftedNeighborhood,
        with_defects = False):

    if with_defects:
        assert False, "Currently not supported"
    print "Computing multcut features for lifted neighborhood", liftedNeighborhood

    from concurrent import futures

    # variables for the multicuts
    uv_ids_local     = ds._adjacent_segments(segId)
    seg_id_max = ds.seg(segId).max()
    n_var = seg_id_max + 1

    # scaling for energies
    edge_probs = pLocal.copy()

    edge_indications = ds.edge_indications(segId)
    edge_areas       = ds._rag(segId).edgeLengths()

    def single_mc(pLocal, edge_indications, edge_areas,
            uv_ids, n_var, seg_id, pipelineParam, beta, weight):

        # copy the probabilities
        probs = pLocal.copy()
        p_min = 0.001
        p_max = 1. - p_min
        probs = (p_max - p_min) * edge_probs + p_min
        # probs to energies
        energies = np.log( (1. - probs) / probs ) + np.log( (1. - beta) / beta )

        # weight the energies
        if pipelineParam.weighting_scheme == "z":
            print "Weighting Z edges"
            # z - edges are indicated with 0 !
            area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
            # we only weight the z edges !
            w = weight * edge_areas[edge_indications == 0] / area_z_max
            energies[edge_indications == 0] = np.multiply(w, energies[edge_indications == 0])

        elif pipelineParam.weighting_scheme == "xyz":
            print "Weighting xyz edges"
            # z - edges are indicated with 0 !
            area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
            len_xy_max = float( np.max( edge_areas[edge_indications == 1] ) )
            # weight the z edges !
            w_z = weight * edge_areas[edge_indications == 0] / area_z_max
            energies[edge_indications == 0] = np.multiply(w_z, energies[edge_indications == 0])
            # weight xy edges
            w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
            energies[edge_indications == 1] = np.multiply(w_xy, energies[edge_indications == 1])

        elif pipelineParam.weighting_scheme == "all":
            print "Weighting all edges"
            area_max = float( np.max( edge_areas ) )
            w = weight * edge_areas / area_max
            energies = np.multiply(w, energies)

        # get the energies (need to copy code here, because we can't use caching in threads)
        mc_node, mc_energy, t_inf = multicut_fusionmoves(
                n_var, uv_ids,
                energies, pipelineParam)

        return mc_node

    # serial for debugging
    #mc_nodes = []
    #for beta in (0.4, 0.45, 0.5, 0.55, 0.65):
    #    for w in (12, 16, 25):
    #        res = single_mc( pLocal, edge_indications, edge_areas,
    #            uv_ids_local, n_var, segId, pipelineParam, beta, w )
    #        mc_nodes.append(res)

    # parralel
    with futures.ThreadPoolExecutor(max_workers=pipelineParam.n_threads) as executor:
        tasks = []
        for beta in (0.4, 0.45, 0.5, 0.55, 0.65):
            for w in (12, 16, 25):
                tasks.append( executor.submit( single_mc, pLocal, edge_indications, edge_areas,
                    uv_ids_local, n_var, segId, pipelineParam, beta, w ) )

    mc_nodes = [future.result() for future in tasks]

    # map multicut result to lifted edges
    allFeat = [ ( mc_node[uvIds[:, 0]] !=  mc_node[uvIds[:, 1]] )[:,None] for mc_node in mc_nodes]

    mcStates = numpy.concatenate(allFeat, axis=1)
    stateSum = numpy.sum(mcStates,axis=1)
    return numpy.concatenate([mcStates,stateSum[:,None]],axis=1)


# TODO adapt to defects
@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_pmap_multicut(ds,
        segId,
        pLocal,
        pipelineParam,
        uvIds,
        liftedNeighborhood,
        with_defects = False):

    if with_defects:
        assert False, "Currently not supported"

    print "Computing multcut features for lifted neighborhood", liftedNeighborhood

    # variables for the multicuts
    uv_ids_local     = ds._adjacent_segments(segId)
    seg_id_max = ds.seg(segId).max()
    n_var = seg_id_max + 1

    # scaling for energies
    edge_probs = pLocal.copy()

    edge_indications = ds.edge_indications(segId)
    edge_areas       = ds._rag(segId).edgeLengths()

    weight = pipelineParam.weight
    beta = pipelineParam.beta_local

    # copy the probabilities
    probs = pLocal.copy()
    p_min = 0.001
    p_max = 1. - p_min
    probs = (p_max - p_min) * edge_probs + p_min
    # probs to energies
    energies = np.log( (1. - probs) / probs ) + np.log( (1. - beta) / beta )

    # weight the energies
    if pipelineParam.weighting_scheme == "z":
        print "Weighting Z edges"
        # z - edges are indicated with 0 !
        area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
        # we only weight the z edges !
        w = weight * edge_areas[edge_indications == 0] / area_z_max
        energies[edge_indications == 0] = np.multiply(w, energies[edge_indications == 0])

    elif pipelineParam.weighting_scheme == "xyz":
        print "Weighting xyz edges"
        # z - edges are indicated with 0 !
        area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
        len_xy_max = float( np.max( edge_areas[edge_indications == 1] ) )
        # weight the z edges !
        w_z = weight * edge_areas[edge_indications == 0] / area_z_max
        energies[edge_indications == 0] = np.multiply(w_z, energies[edge_indications == 0])
        # weight xy edges
        w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
        energies[edge_indications == 1] = np.multiply(w_xy, energies[edge_indications == 1])

    elif pipelineParam.weighting_scheme == "all":
        print "Weighting all edges"
        area_max = float( np.max( edge_areas ) )
        w = weight * edge_areas / area_max
        energies = np.multiply(w, energies)

    # compute map
    ret, mc_energy, t_inf, obj = multicut_fusionmoves(n_var,
            uv_ids_local,
            energies,
            pipelineParam,
            returnObj=True)

    ilpFactory = obj.multicutIlpFactory(ilpSolver='cplex',
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
        #memLimit= 0.01
    )
    print "Map solution computed starting perturb"
    greedy = obj.greedyAdditiveFactory()

    fmFactory = obj.fusionMoveBasedFactory(
        fusionMove=obj.fusionMoveSettings(mcFactory=greedy),
        proposalGen=obj.watershedProposals(sigma=1,seedFraction=0.01),
        numberOfIterations=100,
        numberOfParallelProposals=16, # no effect if nThreads equals 0 or 1
        numberOfThreads=0,
        stopIfNoImprovement=20,
        fuseN=2,
    )
    s = obj.perturbAndMapSettings(mcFactory=fmFactory,noiseMagnitude=4.2,numberOfThreads=-1,
                            numberOfIterations=pipelineParam.pAndMapIterations, verbose=2, noiseType='uniform')
    pAndMap = obj.perturbAndMap(obj, s)
    pmapProbs =  pAndMap.optimize(ret)

    # use perturb and map probs with clustering
    return clusteringFeatures(ds, segId,
            uvIds, pmapProbs, pipelineParam.lifted_neighborhood, True )


def lifted_feature_aggregator(ds,
        trainsets,
        featureList,
        featureListLocal,
        pLocal,
        pipelineParam,
        uvIds,
        segId,
        with_defects = False):

    assert len(featureList) > 0
    for feat in featureList:
        assert feat in ("mc", "cluster","reg","multiseg","perturb"), feat

    features = []
    if "mc" in featureList:# TODO make defect proof
        features.append(
                compute_lifted_feature_multicut(ds,
                    segId,
                    pLocal,
                    pipelineParam,
                    uvIds,
                    pipelineParam.lifted_neighborhood,
                    with_defects) )
    if "perturb" in featureList:# TODO make defect proof
        features.append(
                compute_lifted_feature_pmap_multicut(ds,
                    segId,
                    pLocal,
                    pipelineParam,
                    uvIds,
                    pipelineParam.lifted_neighborhood,
                    with_defects) )
    if "cluster" in featureList:
        features.append(
                clusteringFeatures(ds,
                    segId,
                    uvIds,
                    pLocal,
                    pipelineParam.lifted_neighborhood,
                    False,
                    with_defects) )
    if "reg" in featureList: # this should be defect proof without any adjustments!
        features.append(
                ds.region_features(segId,
                    0,
                    uvIds,
                    pipelineParam.lifted_neighborhood) )
    if "multiseg" in featureList:# TODO reactivate
        features.append(
                compute_lifted_feature_multiple_segmentations(ds,
                    trainsets,
                    segId,
                    featureListLocal,
                    uvIds,
                    pipelineParam) )
    if exp_params.use_2d: # lfited distance as extra feature if we use extra features for 2d edges
        nz_train = node_z_coord(ds, segId)
        lifted_distance = np.abs(
                np.subtract( nz_train[uv_ids_train[:,0]],
                        nz_train[uv_ids_train[:,1]]) )
        features.append(lifted_distance[:,None])

    return numpy.concatenate( features, axis=1 )


@cacher_hdf5()
def compute_and_save_lifted_nh(ds,
        segId,
        liftedNeighborhood,
        with_defects = False):

    if with_defects:
        n_nodes, uvs_local = modified_mc_problem(ds, segId)
    else:
        rag = ds._rag(segId)
        n_nodes = rag.nodeNum
        uvs_local = rag.uvIds()

    originalGraph = agraph.Graph(n_nodes)
    originalGraph.insertEdges(uvs_local)

    print ds.ds_name
    print "Computing lifted neighbors for range:", liftedNeighborhood
    lm = agraph.liftedMcModel(originalGraph)
    agraph.addLongRangeNH(lm , liftedNeighborhood)
    uvIds = lm.liftedGraph().uvIds()

    return uvIds[uvs_local.shape[0]:,:]


# TODO adapt to defects
# we assume that uv is consecutive
#@cacher_hdf5()
# sample size 0 means we do not sample!
def compute_and_save_long_range_nh(uvIds, min_range, max_sample_size=0):
    import random
    import itertools

    originalGraph = agraph.Graph(uvIds.max()+1)
    originalGraph.insertEdges(uvIds)

    uv_long_range = np.array(list(itertools.combinations(np.arange(originalGraph.numberOfVertices), 2)), dtype=np.uint64)

    lm_short = agraph.liftedMcModel(originalGraph)
    agraph.addLongRangeNH(lm_short, min_range)
    uvs_short = lm_short.liftedGraph().uvIds()

    # Remove uvs_short from uv_long_range
    # -----------------------------------

    # Concatenate both lists
    concatenated = np.concatenate((uvs_short, uv_long_range), axis=0)

    # Find unique rows according to
    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = np.ascontiguousarray(concatenated).view(np.dtype((np.void, concatenated.dtype.itemsize * concatenated.shape[1])))
    uniques, idx, counts = np.unique(b, return_index=True, return_counts=True)

    # Extract those that have count == 1
    # TODO this is not tested
    long_range_idx = idx[counts == 1]
    uv_long_range = concatenated[long_range_idx]

    # Extract random sample
    if max_sample_size:
        sample_size = min(max_sample_size, uv_long_range.shape[0])
        uv_long_range = np.array(random.sample(uv_long_range, sample_size))

    return uv_long_range


@cacher_hdf5()
def node_z_coord(ds, segId):
    rag = ds._rag(segId)
    labels = rag.labels
    labels = labels.squeeze()
    nz = numpy.zeros(rag.maxNodeId +1, dtype='uint32')
    for z in range(labels.shape[2]):
        lz = labels[:,:,z]
        nz[lz] = z
    return nz


@cacher_hdf5(ignoreNumpyArrays=True)
def lifted_fuzzy_gt(ds, segId, uvIds):
    if ds.has_seg_mask:
        assert False, "Fuzzy gt not supported yet for segmentation mask"
    gt = ds.gt()
    oseg = ds.seg(segId)
    fuzzyLiftedGt = agraph.candidateSegToRagSeg(
    oseg.astype('uint32'), gt.astype('uint32'),
        uvIds.astype(np.uint64))
    return fuzzyLiftedGt


@cacher_hdf5(ignoreNumpyArrays=True)
def lifted_hard_gt(ds, segId, uvIds):
    rag = ds._rag(segId)
    gt = ds.gt()
    nodeGt,_ =  rag.projectBaseGraphGt(gt)
    labels   = (nodeGt[uvIds[:,0]] != nodeGt[uvIds[:,1]])
    return labels


def mask_lifted_edges(ds,
        seg_id,
        labels,
        uv_ids,
        exp_params):

    labeled = np.ones_like(labels, bool)

    # mask edges in ignore mask
    if exp_params.use_ignore_mask:
        ignore_mask = ds_train.lifted_ignore_mask(
            seg_id,
            exp_params.lifted_neighborhood,
            uv_ids)
        labeled[ignore_mask] = False

    # check which of the edges is in plane and mask the others
    if exp_params.learn_2d:
        nz_train = node_z_coord(ds, seg_id)
        zU = nz_train[uv_ids[:,0]]
        zV = nz_train[uv_ids[:,1]]
        if with_defects:
            raise AttributeError("2d learning not supported for pipeline with defect correction, yet")
        ignore_mask = (zU != zV)
        labeled[ignore_mask] = False

    # ignore all edges that are connected to the ignore label (==0) in the seg mask
    if ds.has_seg_mask:
        ignore_mask = (uv_ids == 0).any(axis = 1)
        assert ignore_mask.shape[0] == labels.shape[0]
        labeled[ ignore_mask ] = False

    return labeled


def learn_lifted_rf(cache_folder,
        trainsets,
        seg_id,
        feature_list_lifted,
        feature_list_local,
        exp_params,
        trainstr,
        paramstr,
        with_defects = False):

    # check if already cached
    if cache_folder is not None: # we use caching for the rf => look if already exists
        rf_folder = os.path.join(cache_folder, "lifted_rf" + trainstr)
        rf_name = "rf_" + "_".join( [trainstr, paramstr] ) + ".h5"
        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)
        rf_path   = os.path.join(rf_folder, rf_name)
        if os.path.exists(rf_path):
            return RandomForest(rf_path, 'rf')
            #if with_defects: TODO figure out if we need different random forests
            #    return RandomForest(rf_path, 'rf'), RandomForest(rf_path, 'rf_defects')

    features_train = []
    labels_train   = []

    for ds_train in trainsets:

        assert ds_train.n_cutouts == 3, "Wrong number of cutouts: " + str(ds_train.n_cutouts)
        train_cut = ds_train.get_cutout(1)

        # get edge probabilities from random forest on training set cut out in the middle
        p_local_train = learn_and_predict_rf_from_gt(
            exp_params.rf_cache_folder,
            [ds_train.get_cutout(0), ds_train.get_cutout(2)],
            train_cut,
            seg_id_train,
            seg_id_train,
            feature_list_local,
            exp_params,
            with_defects)

        uv_ids_train = compute_and_save_lifted_nh(
            train_cut,
            seg_id_train,
            exp_params.lifted_neighborhood,
            with_defects)

        # compute the features for the training set
        f_train = lifted_feature_aggregator(
            train_cut,
            [ds_train.get_cutout(0), ds_train.get_cutout(2)],
            feature_list_lifted,
            feature_list_local,
            p_local_train,
            exp_params,
            uv_ids_train,
            seg_id_train,
            with_defects)

        labels = lifted_hard_gt(train_cut, seg_id_train, uv_ids_train)

        labeled = mask_lifted_edges(train_cut,
                seg_id_train,
                labels,
                uv_ids_train,
                exp_params)

        features_train.append(f_train[labeled])
        labels_train.append(labels[labeled])

    features_train = np.concatenate(features_train, axis = 0)
    labels_train = np.concatenate(labels_train, axis = 0)

    rf = RandomForest(features_train.astype('float32'),
            labels_train.astype('uint32'),
            treeCount = exp_params.n_trees,
            n_threads = exp_params.n_threads,
            max_depth = 10 )

    if cache_folder is not None:
        rf.writeHDF5(rf_path, 'rf')
    return rf


def learn_and_predict_lifted_rf(trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list_lifted, feature_list_local,
        exp_params, with_defects = False):

    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list), type(trainsets)
    if not isinstance(trainsets, list):
        trainsets = [trainsets,]

    # strings for caching
    # str for all relevant params
    paramstr = "_".join( ["_".join(lifted_feature_list), "_".join(local_feature_list),
        str(exp_params.anisotropy_factor), str(exp_params.learn_2d),
        str(exp_params.use_2d), str(exp_params.lifted_neighborhood),
        str(exp_params.use_ignore_mask), str(with_defects)] )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets ]) + "_" + str(seg_id_train)

    uv_ids_test = compute_and_save_lifted_nh(ds_test,
            seg_id_test,
            exp_params.lifted_neighborhood,
            with_defects)
    nz_test = node_z_coord(ds_test, seg_id_test)

    if cache_folder is not None: # cache-folder exists => look if we already have a prediction
        pred_folder = os.path.join(cache_folder, "lifted_prediction_" + trainstr)
        pred_name = "prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"
        if with_defects:
            pred_name =  pred_name[:-3] + "_with_defects.h5"
        if len(pred_name) >= 256:
            pred_name = str(hash(pred_name[:-3])) + ".h5"
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)
        pred_path = os.path.join(pred_folder, pred_name)
        # see if the rf is already learned and predicted, otherwise learn it
        if os.path.exists(pred_path):
            return vigra.readHDF5(pred_path, 'data'), uv_ids_test, nz_test

    rf = learn_lifted_rf(cache_folder,
        trainsets,
        seg_id_train,
        feature_list_lifted,
        feature_list_local,
        exp_params,
        trainstr,
        paramstr,
        with_defects)

    # get edge probabilities from random forest on test set
    p_local_test = learn_and_predict_rf_from_gt(pipelineParam.rf_cache_folder,
        [ds_train.get_cutout(i) for i in (0,2) for ds_train in trainsets], ds_test,
        seg_id_train, seg_id_test,
        feature_list_local, exp_params,
        with_defects)

    features_test = lifted_feature_aggregator(ds_test,
            [ds_train.get_cutout(i) for i in (0,2) for ds_train in trainsets],
            feature_list_lifted, feature_list_local,
            p_local_test, exp_params,
            uv_ids_test, seg_id_test,
            with_defects)

    p_test = rf.predictProbabilities(
            features_test.astype('float32'),
            n_threads = pipelineParam.n_threads)[:,1]
    p_test /= rf.treeCount()
    pTest[np.isnan(pTest)] = .5
    assert not np.isnan(pTest).any(), str(np.isnan(pTest).sum())
    if cache_folder is not None:
        vigra.writeHDF5(p_test, pred_path, 'data')

    return p_test, uv_ids_test, nz_test


def optimizeLifted(uvs_local,
        uvs_lifted,
        costs_local,
        costs_lifted,
        starting_point = None):
    print "Optimizing lifted model"

    assert uvs_local.shape[0] == costs_local.shape[0], "Local uv ids and energies do not match!"
    assert uvs_lifted.shape[0] == costs_lifted.shape[0], "Lifted uv ids and energies do not match!"
    n_nodes = uvs_local.max() + 1
    assert n_nodes >= uvs_lifted.max() + 1, "Local and lifted nodes do not match!"

    # build the lifted model
    graph = agraph.Graph(n_nodes)
    graph.insertEdges(uvs_local)
    model = agraph.liftedMcModel(graph)

    # set cost for local edges
    model.setCosts(uvs_local, costs_local)
    # set cost for lifted edges
    model.setCosts(uvs_lifted, costs_lifted)

    # if no starting point is given, start with ehc solver
    if starting_point is None:
        # settings ehc
        print "Starting from ehc solver result"
        settingsGa = agraph.settingsLiftedGreedyAdditive(model)
        ga = agraph.liftedGreedyAdditive(model, settingsGa)
        ws = ga.run()
    # else, we use the starting point that is given as argument
    else:
        print "Starting from external starting point"
        ws = starting_point.astype('uint8')

    # setttings for kl
    settingsKl = agraph.settingsLiftedKernighanLin(model)
    kl = agraph.liftedKernighanLin(model, settingsKl)
    ws2 = kl.run(ws)


    # FM RAND
    # settings for proposal generator
    settingsProposalGen = agraph.settingsProposalsFusionMovesRandomizedProposals(model)
    settingsProposalGen.sigma = 10.5
    settingsProposalGen.nodeLimit = 0.5

    # settings for solver itself
    settings = agraph.settingsFusionMoves(settingsProposalGen)
    settings.maxNumberOfIterations = 1
    settings.nParallelProposals = 2
    settings.reduceIterations = 1
    settings.seed = 42
    settings.verbose = 0
    # solver
    solver = agraph.fusionMoves(model, settings)
    ws3 = solver.run(ws2)


    # FM SG
    # settings for proposal generator
    settingsProposalGen = agraph.settingsProposalsFusionMovesSubgraphProposals(model)
    settingsProposalGen.subgraphRadius = 5

    # settings for solver itself
    settings = agraph.settingsFusionMoves(settingsProposalGen)
    settings.maxNumberOfIterations = 3
    settings.nParallelProposals = 10
    settings.reduceIterations = 0
    settings.seed = 43
    settings.verbose = 0
    # solver
    solver = agraph.fusionMoves(model, settings)
    out = solver.run(ws3)

    nodeLabels = model.edgeLabelsToNodeLabels(out)
    nodeLabels = nodeLabels.astype('uint32')

    return nodeLabels
