import vigra
import os
import cPickle as pickle

import vigra.graphs as vgraph

import graph as agraph
#import nifty
import numpy
#import scipy.ndimage
from MCSolverImpl import *
from Tools import *
from sklearn.ensemble import RandomForestClassifier



def hessianEv(img, sigma):
    if img.squeeze().ndim  == 2:
        return vigra.filters.hessianOfGaussianEigenvalues(img, sigma)[:,:,0]
    else:
        out = numpy.zeros(img.shape, dtype='float32')
        for z in range(img.shape[2]):
            out[:, :, z] = vigra.filters.hessianOfGaussianEigenvalues(img[:,:,z], sigma)[:,:,0]
        return out



def gaussianSmooth(img, sigma):
    if img.squeeze().ndim  == 2:
        return vigra.filters.gaussianSmoothing(img, sigma)
    else:
        out = numpy.zeros(img.shape, dtype='float32')
        for z in range(img.shape[2]):
            out[:, :, z] = vigra.filters.gaussianSmoothing(img[:,:,z], sigma)
        return out




@cacher_hdf5(ignoreNumpyArrays=True)
def clusteringFeatures(ds, segId, extraUV, edgeIndicator, liftedNeighborhood, is_perturb_and_map ):

    print "Computing clustering features for lifted neighborhood", liftedNeighborhood
    if is_perturb_and_map:
        print "For perturb and map"
    else:
        print "For normal clustering"

    originalGraph =  ds._rag(segId)
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


# TODO we might even loop over different solver here ?! -> nifty greedy ?!
@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_multiple_segmentations(ds, trainsets, referenceSegId, featureListLocal, uvIdsLifted, pipelineParam):

    import nifty

    print "Computing lifted features from multople segmentations from %i segmentations for reference segmentation %i" % (ds.n_seg, referenceSegId)

    from concurrent import futures

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

        mc_node, mc_energy, t_inf = nifty_fusionmoves(
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



@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_multicut(ds, segId, pLocal, pipelineParam, uvIds, liftedNeighborhood):

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
        mc_node, _, mc_energy, t_inf = multicut_fusionmoves(
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




@cacher_hdf5(ignoreNumpyArrays=True)
def compute_lifted_feature_pmap_multicut(ds, segId, pLocal, pipelineParam, uvIds, liftedNeighborhood):

    import nifty

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
    ret, mc_energy, t_inf, obj = nifty_fusionmoves(n_var, uv_ids_local, energies, pipelineParam, returnObj=True)

    ilpFactory = obj.multicutIlpFactory(ilpSolver='cplex',
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
        #memLimit= 0.01
    )
    print "Map solution computed starting perturb"
    greedy = obj.greedyAdditiveFactory()

    fmFactory = obj.fusionMoveBasedFactory(
        fusionMove=obj.fusionMoveSettings(mcFactory=greedy),
        #fusionMove=obj.fusionMoveSettings(mcFactory=ilpFactory),
        #proposalGen=nifty.greedyAdditiveProposals(sigma=30,nodeNumStopCond=-1,weightStopCond=0.0),
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
    #print pmapProbs


    # use perturb and map probs with clustering

    return clusteringFeatures(ds, segId,
            uvIds, pmapProbs, pipelineParam.lifted_neighborhood, True )


def lifted_feature_aggregator(ds, trainsets, featureList, featureListLocal,
        pLocal, pipelineParam, uvIds, segId ):

    assert len(featureList) > 0
    for feat in featureList:
        assert feat in ("mc", "cluster","reg","multiseg","perturb"), feat

    features = []
    if "mc" in featureList:
        features.append( compute_lifted_feature_multicut(ds, segId,
            pLocal, pipelineParam, uvIds, pipelineParam.lifted_neighborhood) )
    if "perturb" in featureList:
        features.append( compute_lifted_feature_pmap_multicut(ds, segId,
            pLocal, pipelineParam, uvIds, pipelineParam.lifted_neighborhood) )
    if "cluster" in featureList:
        features.append( clusteringFeatures(ds, segId,
            uvIds, pLocal, pipelineParam.lifted_neighborhood, False ) )
    if "reg" in featureList:
        features.append( ds.region_features(segId, 0, uvIds, pipelineParam.lifted_neighborhood) )
    if "multiseg" in featureList:
        features.append( compute_lifted_feature_multiple_segmentations(ds, trainsets, segId, featureListLocal, uvIds, pipelineParam) )

    for feat in features:
        print feat.shape

    return numpy.concatenate( features, axis=1 )


@cacher_hdf5()
def compute_and_save_lifted_nh(ds, segId, liftedNeighborhood):

    rag = ds._rag(segId)
    originalGraph = agraph.Graph(rag.nodeNum)
    originalGraph.insertEdges(rag.uvIds())

    print ds.ds_name
    print "Computing lifted neighbors for range:", liftedNeighborhood
    lm= agraph.liftedMcModel(originalGraph)
    agraph.addLongRangeNH(lm , liftedNeighborhood)
    uvIds = lm.liftedGraph().uvIds()

    return uvIds[rag.edgeNum:,:]


# we assume that uv is consecutive
#@cacher_hdf5()
def compute_and_save_long_range_nh(uvIds, min_range):

    originalGraph = agraph.Graph(uvIds.max()+1)
    originalGraph.insertEdges(uvIds)

    import itertools
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
    # uniques = concatenated[idx]

    # Extract those that have count == 1
    uv_long_range = uniques[counts == 1].view(concatenated.dtype)
    uv_long_range = uv_long_range.reshape((uv_long_range.shape[0]/2, 2))

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

    gt = ds.gt()
    oseg = ds.seg(segId)


    fuzzyLiftedGt = agraph.candidateSegToRagSeg(
    oseg.astype('uint32'), gt.astype('uint32'),
        uvIds.astype(np.uint64))

    return fuzzyLiftedGt

#@cacher_hdf5(ignoreNumpyArrays=True)
def lifted_hard_gt(ds, segId, uvIds):

    rag = ds._rag(segId)
    gt = ds.gt()
    nodeGt,_ =  rag.projectBaseGraphGt(gt)
    labels   = (nodeGt[uvIds[:,0]] != nodeGt[uvIds[:,1]]).astype('float32')

    return labels, nodeGt


# TODO we should cache this for rerunning experiments
# (will take very long for larger ds like snemi)
#@cacher_hdf5(ignoreNumpyArrays=True)
def doActualTrainingAndPrediction(trainSets, dsTest, X, Y, F, pipelineParam, oob = False):

    if pipelineParam.verbose:
        verbose = 2
    else:
        verbose = 0

    rf_save_folder = os.path.join(pipelineParam.rf_cache_folder,"lifted_rfs" )
    if not os.path.exists(rf_save_folder):
        os.mkdir(rf_save_folder)

    rf_save_path = os.path.join(rf_save_folder,
            "rf_trainsets_" + "_".join([ds.ds_name for ds in trainSets]) + ".pkl")

    pred_save_folder = os.path.join(rf_save_folder, "predictions")
    if not os.path.exists(pred_save_folder):
        os.mkdir(pred_save_folder)
    pred_save_path = os.path.join(pred_save_folder,
            "pred_trainsets_" + "_".join([ds.ds_name for ds in trainSets]) + "_testset_" + dsTest.ds_name + ".h5")

    if os.path.exists(pred_save_path):
        return vigra.readHDF5(pred_save_path, 'data')

    else:

        if os.path.exists(rf_save_path):
            with open(rf_save_path,'r') as f:
                rf = pickle.load(f)

        else:
            rf = RandomForestClassifier(n_estimators = pipelineParam.n_trees,
                n_jobs=pipelineParam.n_threads_lifted, oob_score=oob, verbose = verbose,
                min_samples_leaf = 10, max_depth = 10 )
            rf.fit(X, Y.astype('uint32'))
            print "Trained RF on lifted edges:"
            if oob:
                print "OOB-Error:", 1. - rf.oob_score_
            with open(rf_save_path,'w') as f:
                pickle.dump(rf,f)

        pTest = rf.predict_proba(F)[:,1]
        vigra.writeHDF5(pTest,pred_save_path,'data')
        return pTest



def learn_and_predict_lifted(trainsets, dsTest,
        segIdTrain, segIdTest,
        feature_list_lifted, feature_list_local,
        pipelineParam ):

    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list), type(trainsets)

    if not isinstance(trainsets, list):
        trainsets = [trainsets,]

    featuresTrain = []
    labelsTrain   = []

    uvIdsTest = compute_and_save_lifted_nh(dsTest, segIdTest, pipelineParam.lifted_neighborhood)
    nzTest  = node_z_coord(dsTest, segIdTest)
    for dsTrain in trainsets:

        # get edge probabilities from random forest on training set cut out in the middle
        pLocalTrain = learn_and_predict_rf_from_gt(
            pipelineParam.rf_cache_folder,
            [dsTrain.get_cutout(0), dsTrain.get_cutout(2)],
            dsTrain.get_cutout(1) ,
            segIdTrain,
            segIdTrain,
            feature_list_local,
            pipelineParam)

        uvIdsTrain = compute_and_save_lifted_nh(
            dsTrain.get_cutout(1),
            segIdTrain,
            pipelineParam.lifted_neighborhood)

        nzTrain = node_z_coord(dsTrain.get_cutout(1), segIdTrain)

        # compute the features for the training set
        fTrain = lifted_feature_aggregator(
            dsTrain.get_cutout(1),
            [dsTrain.get_cutout(0), dsTrain.get_cutout(2)],
            feature_list_lifted,
            feature_list_local,
            pLocalTrain,
            pipelineParam,
            uvIdsTrain,
            segIdTrain)

        dsTrain = dsTrain.get_cutout(1)
        labels, nodeGt = lifted_hard_gt(dsTrain, segIdTrain, uvIdsTrain)

        if pipelineParam.use_ignore_mask:
            ignoreMask = dsTrain.lifted_ignore_mask(
                segIdTrain,
                pipelineParam.lifted_neighborhood,
                uvIdsTrain)
            labels[ignoreMask] = 0.5

        # check which of the edges is in plane
        zU = nzTrain[uvIdsTrain[:,0]]
        zV = nzTrain[uvIdsTrain[:,1]]

        #where in plane
        if pipelineParam.learn_2d:
            ignoreMask = (zU != zV)
            labels[ignoreMask] = 0.5

        labeled = labels != 0.5

        X =  fTrain[labeled,:]
        labels = labels[labeled].astype('uint8')

        featuresTrain.append(X)
        labelsTrain.append(labels)

        #YF = fuzzyLiftedGt[whereGt]
        #Y = numpy.zeros(YF.shape)
        #Y[YF>0.5] = 1

    featuresTrain = np.concatenate(featuresTrain, axis = 0)
    labelsTrain = np.concatenate(labelsTrain, axis = 0)

    # get edge probabilities from random forest on test set
    pLocalTest = learn_and_predict_rf_from_gt(pipelineParam.rf_cache_folder,
        [dsTrain.get_cutout(i) for i in (0,2) for dsTrain in trainsets], dsTest,
        segIdTrain, segIdTest,
        feature_list_local, pipelineParam)

    fTest = lifted_feature_aggregator(dsTest,
            [dsTrain.get_cutout(i) for i in (0,2) for dsTrain in trainsets],
                feature_list_lifted, feature_list_local,
                pLocalTest, pipelineParam, uvIdsTest, segIdTest)

    pTest = doActualTrainingAndPrediction(trainsets, dsTest, featuresTrain, labelsTrain, fTest, pipelineParam)

    #featuresTrain = np.zeros(10)
    #labelsTrain = np.zeros(10)
    #fTest = np.zeros(10)
    #pTest = doActualTrainingAndPrediction(dsTest, featuresTrain, labelsTrain, fTest, pipelineParam)

    return pTest, uvIdsTest, nzTest




def optimizeLifted(dsTest, model, rag, starting_point = None):
    print "Optimizing lifted model"

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
