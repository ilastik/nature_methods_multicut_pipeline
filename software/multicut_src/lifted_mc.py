import os

# TODO numpy -> np to be consistent with the rest of the ppl
import numpy
import vigra

import vigra.graphs as vgraph
import graph as agraph

from concurrent import futures

from DataSet import DataSet
from MCSolverImpl import multicut_fusionmoves
from tools import cacher_hdf5
from EdgeRF import learn_and_predict_rf_from_gt, RandomForest
from MCSolverImpl import weight_z_edges, weight_all_edges, weight_xyz_edges

from defect_handling import defects_to_nodes, find_matching_indices, modified_adjacency, modified_topology_features


# TODO this is quite the bottleneck, speed up !
# returns indices of lifted edges that are ignored due to defects
@cacher_hdf5(ignoreNumpyArrays=True)
def lifted_ignore_ids(ds,
        seg_id,
        uv_ids):
    defect_nodes = defects_to_nodes(ds, seg_id)
    return find_matching_indices(uv_ids, defect_nodes)


# TODO use nifty agglomertion
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

    uvs_local = modified_adjacency(ds, segId) if (with_defects and ds.has_defects) else ds._adjacent_segments(segId)
    n_nodes = uvs_local.max() + 1

    # if we have a segmentation mask, remove all the uv ids that link to the ignore segment (==0)
    if ds.has_seg_mask:
        where_uv_local = (uvs_local != ds.ignore_seg_value).all(axis = 1)
        uvs_local      = uvs_local[where_uv_local]
        edgeIndicator  = edgeIndicator[where_uv_local]
        assert numpy.sum( (extraUV == ds.ignore_seg_value).any(axis = 1) ) == 0
    assert edgeIndicator.shape[0] == uvs_local.shape[0]

    originalGraph = vgraph.listGraph(n_nodes)
    originalGraph.addEdges(uvs_local)

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

    eLen = vgraph.getEdgeLengths(originalGraph)
    nodeSizes_ = vgraph.getNodeSizes(originalGraph)

    # FIXME GIL is not lifted for vigra function (probably cluster)
    def cluster(wardness):

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

        assert mg.edgeNum == 0, str(mg.edgeNum)

        # FIXME I am disabling these checks for now, but will need to investigate this further
        # They can fail because with defects and seg mask we can get unconnected pieces in the graph

        # if we have completely defected slcies, we get a non-connected merge graph
        # TODO I don't know if this is a problem, if it is, we could first remove them
        # and then add dummy values later
        #if not with_defects:
        #    assert mg.nodeNum == 1, str(mg.nodeNum)
        #else:
        #    # TODO need list of defected slices
        #    # TODO test hypothesis
        #    assert mg.nodeNum == len(defect_slices) + 1, "%i, %i" % (mg.nodeNum, len(defect_slices) + 1)

        tweight = edgeIndicatorNew.copy()
        hc.ucmTransform(tweight)

        whereInLifted = liftedGraph.findEdges(extraUV)
        assert whereInLifted.min() >= 0
        feat = tweight[whereInLifted]
        assert feat.shape[0] == extraUV.shape[0]
        return  feat[:,None]

    wardness_vals = [0.01, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7]
    # TODO set from ppl parameter
    with futures.ThreadPoolExecutor(max_workers = 8) as executor:
        tasks = [executor.submit(cluster, w) for w in wardness_vals]
        allFeat = [t.result() for t in tasks]

    weights = numpy.concatenate(allFeat,axis=1)
    mean = numpy.mean(weights,axis=1)[:,None]
    stddev = numpy.std(weights,axis=1)[:,None]
    allFeat = numpy.nan_to_num(
            numpy.concatenate([weights,mean,stddev],axis=1) )
    allFeat = numpy.require(allFeat, dtype = 'float32')
    assert allFeat.shape[0] == extraUV.shape[0]

    return allFeat

#
# Features from ensembling over segmentations
#

# TODO common ensembling backend
# TODO adapt for defects
# TODO also use similar feature for ucm
#@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_mala_agglomeration(
        ds,
        seg_id,
        inp_ids,
        uv_ids_lifted,
        lifted_nh,
        with_defects = False
        ):

    assert len(inp_ids) == 2
    print "Computing mala agglomeration features for lifted neighborhood", lifted_nh
    assert not with_defects, "Not Implemented for defects yet!"

    rag = ds._rag(seg_id)
    edge_indications = ds.edge_indications(seg_id)
    edge_lens        = ds.topology_features(seg_id, False)[:,0]

    # get the max affinities for xy edges from xy affinities
    aff_xy = ds.inp(inp_ids[0])
    edge_map_xy  = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, aff_xy)
    indicators = rag.accumulateEdgeStatistics(edge_map_xy)[:,3] # 3 -> max

    # get the max affinities for z edges from z affinities
    aff_z  = ds.inp(inp_ids[1])
    edge_map_z  = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, aff_z)
    indicators_z = rag.accumulateEdgeStatistics(edge_map_z)[:,3] # 3 -> max

    # merge the indicators for xy and z edges
    indicators[edge_indications==1] = indicators_z[edge_indications==1]

    def agglomerate(threshold, use_edge_len):
        graph = nifty.graph.UndirectedGraph(rag.numberOfNodes)
        graph.insertEdges(ds._adjacent_segments(seg_id))

        policy = nifty.graph.agglo.malaClusterPolicy(
                graph = graph,
                edgeIndicators = indicators,
                edgeSizes = edge_lens if use_edge_len else np.ones(graph.numberOfEdges),
                nodeSizes = np.ones(graph.numberOfNodes),
                threshold = threshold)

        clustering = nifty.graph.agglo.agglomerativeClustering(policy)
        clustering.run()

        clustered_nodes = clustering.result()

    # TODO set n workers from ppl params
    with ThreadPoolExecutor(max_workers = 8) as executor:
        tasks = []
        for use_edge_len in (True, False):
            for threshold in (.3,.4,.5,.6,.7,.8):
                tasks.append(executor.submit(agglomerate, use_edge_len, threshold))

    node_results = [t.result() for t in tasks]

    # map multicut result to lifted edges
    edge_results = numpy.concatenate(
            [ ( node_res[uv_ids_lifted[:, 0]] != node_res[uv_ids_lifted[:, 1]] )[:,None] for node_res in node_results],
            axis = 1)
    state_sum = numpy.sum(edge_results, axis=1)
    return numpy.concatenate([edge_results, state_sum[:,None]], axis=1)


# TODO unify with mala ensembling (common function for ensembling over the segmentations)
@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_multicut(
        ds,
        seg_id,
        pmap_local,
        exp_params,
        uv_ids_lifted,
        lifted_nh,
        with_defects = False):

    print "Computing multicut features for lifted neighborhood", lifted_nh
    # variables for the multicuts
    uv_ids_local = modified_adjacency(ds, seg_id) if with_defects and ds.has_defects else ds._adjacent_segments(seg_id)
    n_var = uv_ids_local.max() + 1
    assert pmap_local.shape[0] == uv_ids_local.shape[0]

    edge_indications = modified_edge_indications(ds, seg_id) if with_defects and ds.has_defects else ds.edge_indications(seg_id)
    edge_areas       = modified_topology_features(ds, seg_id, False)[:,0] if with_defects and ds.has_defects else ds.topology_features(seg_id, False)[:,0]

    # set ignore edges to be maximally repulsive
    if with_defects:
        ignore_defect_edge_ids = get_ignore_edge_ids(ds, seg_id)

    # set the edges within the segmask to be maximally repulsive
    if ds.has_seg_mask:
        ignore_seg_mask = (uv_ids_local == ds.ignore_seg_value).any(axis = 1)

    def single_mc(beta, weight):
        # copy the probabilities
        costs = pmap_local.copy()
        p_min = 0.001
        p_max = 1. - p_min
        costs = (p_max - p_min) * costs + p_min
        # probs to energies
        costs = numpy.log( (1. - probs) / probs ) + numpy.log( (1. - beta) / beta )

        # weight the energies
        if exp_params.weighting_scheme == "z":
            costs = weight_z_edges(costs, edge_areas, edge_indications, weight)
        elif exp_params.weighting_scheme == "xyz":
            costs = weight_xyz_edges(costs, edge_areas, edge_indications, weight)
        elif exp_params.weighting_scheme == "all":
            costs = weight_all_edges(costs, edge_areas, weight)

        max_repulsive = 2 * costs.min()
        if with_defects and ds.has_defects:
            costs[ignore_defect_edge_ids] = max_repulsive
        if ds.has_seg_mask:
            costs[ignore_seg_mask] = max_repulsive

        # get the energies (need to copy code here, because we can't use caching in threads)
        mc_node, mc_energy, t_inf = multicut_fusionmoves(
                n_var, uv_ids_local,
                costs, exp_params)

        return mc_node

    # serial for debugging
    #mc_nodes = []
    #for beta in (0.4, 0.45, 0.5, 0.55, 0.65):
    #    for w in (12, 16, 25):
    #        res = single_mc( pLocal, edge_indications, edge_areas,
    #            uv_ids_local, n_var, seg_id, exp_params, beta, w )
    #        mc_nodes.append(res)

    # parralel
    with futures.ThreadPoolExecutor(max_workers=exp_params.n_threads) as executor:
        tasks = []
        for beta in (0.4, 0.45, 0.5, 0.55, 0.60):
            for w in (12, 16, 25):
                tasks.append( executor.submit( single_mc, beta, w ) )

    mc_nodes = [future.result() for future in tasks]

    # map multicut result to lifted edges
    allFeat = [ ( mc_node[uv_ids_lifted[:, 0]] !=  mc_node[uv_ids_lifted[:, 1]] )[:,None] for mc_node in mc_nodes]

    mcStates = numpy.concatenate(allFeat, axis=1)
    stateSum = numpy.sum(mcStates,axis=1)
    return numpy.concatenate([mcStates,stateSum[:,None]],axis=1)


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
    # deprecated features
    #for feat in featureList:
    #    assert feat in ("mc", "cluster","reg","multiseg","perturb"), feat
    for feat in featureList:
        assert feat in ("mc", "cluster", "reg", "mala"), feat

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
    if "mala" in featureList:
        features.append(
                compute_lifted_feature_mala_agglomeration(
                    ds,
                    seg_id,
                    (1,2),
                    uv_ids_lifted,
                    pipelineParam.lifted_neighborhood,
                    with_defects)
                )
    if pipelineParam.use_2d: # lfited distance as extra feature if we use extra features for 2d edges
        nz_train = ds.node_z_coord(segId)
        lifted_distance = numpy.abs(
                numpy.subtract(
                        nz_train[uvIds[:,0]],
                        nz_train[uvIds[:,1]]) )
        features.append(lifted_distance[:,None])

    return numpy.concatenate( features, axis=1 )


@cacher_hdf5()
def compute_and_save_lifted_nh(ds,
        segId,
        liftedNeighborhood,
        with_defects = False):

    uvs_local = modified_adjacency(ds, segId) if (with_defects and ds.has_defects) else ds._adjacent_segments(segId)
    n_nodes = uvs_local.max() + 1

    # TODO maybe we should remove the uvs connected to a ignore segment if we have a seg mask
    # should be done if this takes too much time if we have a seg mask
    if ds.has_seg_mask:
        where_uv = (uvs_local != ds.ignore_seg_value).all(axis=1)
        uvs_local = uvs_local[where_uv]

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

    uv_long_range = numpy.array(list(itertools.combinations(numpy.arange(originalGraph.numberOfVertices), 2)), dtype=numpy.uint64)

    lm_short = agraph.liftedMcModel(originalGraph)
    agraph.addLongRangeNH(lm_short, min_range)
    uvs_short = lm_short.liftedGraph().uvIds()

    # Remove uvs_short from uv_long_range
    # -----------------------------------

    # Concatenate both lists
    concatenated = numpy.concatenate((uvs_short, uv_long_range), axis=0)

    # Find unique rows according to
    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = numpy.ascontiguousarray(concatenated).view(numpy.dtype((numpy.void, concatenated.dtype.itemsize * concatenated.shape[1])))
    uniques, idx, counts = numpy.unique(b, return_index=True, return_counts=True)

    # Extract those that have count == 1
    # TODO this is not tested
    long_range_idx = idx[counts == 1]
    uv_long_range = concatenated[long_range_idx]

    # Extract random sample
    if max_sample_size:
        sample_size = min(max_sample_size, uv_long_range.shape[0])
        uv_long_range = numpy.array(random.sample(uv_long_range, sample_size))

    return uv_long_range


@cacher_hdf5(ignoreNumpyArrays=True)
def lifted_fuzzy_gt(ds, segId, uvIds):
    if ds.has_seg_mask:
        assert False, "Fuzzy gt not supported yet for segmentation mask"
    gt = ds.gt()
    oseg = ds.seg(segId)
    fuzzyLiftedGt = agraph.candidateSegToRagSeg(
    oseg.astype('uint32'), gt.astype('uint32'),
        uvIds.astype(numpy.uint64))
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
        exp_params,
        with_defects):

    labeled = numpy.ones_like(labels, bool)

    # mask edges in ignore mask
    if exp_params.use_ignore_mask:
        ignore_mask = ds_train.lifted_ignore_mask(
            seg_id,
            exp_params.lifted_neighborhood,
            uv_ids,
            with_defects)
        labeled[ignore_mask] = False

    # check which of the edges is in plane and mask the others
    if exp_params.learn_2d:
        nz_train = ds.node_z_coord(seg_id)
        zU = nz_train[uv_ids[:,0]]
        zV = nz_train[uv_ids[:,1]]
        ignore_mask = (zU != zV)
        labeled[ignore_mask] = False

    # find all lifted edges that touch a defected node and ignore them
    if with_defects and ds.has_defects:
        labeled[lifted_ignore_ids(ds, seg_id, uv_ids)] = False

    # ignore all edges that are connected to the ignore label (==0) in the seg mask
    # they should all be removed from the lifted edges -> check
    if ds.has_seg_mask:
        ignore_mask = (uv_ids == ds.ignore_seg_value).any(axis = 1)
        assert numpy.sum(ignore_mask) == 0
        #assert ignore_mask.shape[0] == labels.shape[0]
        #labeled[ ignore_mask ] = False

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
            return RandomForest.load_from_file(rf_path, 'rf')

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
            seg_id,
            seg_id,
            feature_list_local,
            exp_params,
            with_defects = with_defects,
            use_2rfs = exp_params.use_2rfs)

        uv_ids_train = compute_and_save_lifted_nh(
            train_cut,
            seg_id,
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
            seg_id,
            with_defects)

        labels = lifted_hard_gt(train_cut, seg_id, uv_ids_train)

        labeled = mask_lifted_edges(train_cut,
                seg_id,
                labels,
                uv_ids_train,
                exp_params,
                with_defects)

        features_train.append(f_train[labeled])
        labels_train.append(labels[labeled])

    features_train = numpy.concatenate(features_train, axis = 0)
    labels_train = numpy.concatenate(labels_train, axis = 0)

    print "Start learning lifted random forest"
    rf = RandomForest(features_train.astype('float32'),
            labels_train.astype('uint32'),
            n_trees = exp_params.n_trees,
            n_threads = exp_params.n_threads,
            max_depth = 10 )

    if cache_folder is not None:
        rf.write(rf_path, 'rf')
    return rf


def learn_and_predict_lifted_rf(cache_folder,
        trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list_lifted, feature_list_local,
        exp_params, with_defects = False):

    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list), type(trainsets)
    if not isinstance(trainsets, list):
        trainsets = [trainsets,]

    # strings for caching
    # str for all relevant params
    paramstr = "_".join( ["_".join(feature_list_lifted), "_".join(feature_list_local),
        str(exp_params.anisotropy_factor), str(exp_params.learn_2d),
        str(exp_params.use_2d), str(exp_params.lifted_neighborhood),
        str(exp_params.use_ignore_mask), str(with_defects)] )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets ]) + "_" + str(seg_id_train)

    uv_ids_test = compute_and_save_lifted_nh(ds_test,
            seg_id_test,
            exp_params.lifted_neighborhood,
            with_defects)
    nz_test = ds_test.node_z_coord(seg_id_test)

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
    p_local_test = learn_and_predict_rf_from_gt(exp_params.rf_cache_folder,
        [ds_train.get_cutout(i) for i in (0,2) for ds_train in trainsets], ds_test,
        seg_id_train, seg_id_test,
        feature_list_local, exp_params,
        with_defects = with_defects,
        use_2rfs = exp_params.use_2rfs)

    features_test = lifted_feature_aggregator(ds_test,
            [ds_train.get_cutout(i) for i in (0,2) for ds_train in trainsets],
            feature_list_lifted, feature_list_local,
            p_local_test, exp_params,
            uv_ids_test, seg_id_test,
            with_defects)

    print "Start prediction lifted random forest"
    p_test = rf.predictProbabilities(
            features_test.astype('float32'),
            n_threads = exp_params.n_threads)[:,1]
    p_test /= rf.treeCount()
    p_test[numpy.isnan(p_test)] = .5
    assert not numpy.isnan(p_test).any(), str(numpy.isnan(p_test).sum())
    if cache_folder is not None:
        vigra.writeHDF5(p_test, pred_path, 'data')

    return p_test, uv_ids_test, nz_test


# TODO use lifted multicut from nifty
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


# TODO weight connections in plane: kappa=20
@cacher_hdf5(ignoreNumpyArrays=True)
def lifted_probs_to_energies(ds,
        edge_probs,
        seg_id,
        edgeZdistance,
        lifted_nh,
        betaGlobal=0.5,
        gamma=1.,
        with_defects = False):

    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    e = numpy.log( (1. - edge_probs) / edge_probs ) + numpy.log( (1. - betaGlobal) / betaGlobal )

    # additional weighting
    e /= gamma

    # weight down the z - edges with increasing distance
    if edgeZdistance is not None:
        assert edgeZdistance.shape[0] == e.shape[0], "%s, %s" % (str(edgeZdistance.shape), str(e.shape))
        e /= (edgeZdistance + 1.)

    uv_ids = compute_and_save_lifted_nh(ds, seg_id, lifted_nh, with_defects)
    # find all lifted edges that touch a defected node and ignore them
    if with_defects and ds.has_defects:
        max_repulsive = 2 * e.min()
        e[lifted_ignore_ids(ds, seg_id, uv_ids)] = max_repulsive

    # set the edges within the segmask to be maximally repulsive
    # these should all be removed, check !
    if ds.has_seg_mask:
        assert numpy.sum((uv_ids == ds.ignore_seg_value).any(axis = 1)) == 0

    return e
