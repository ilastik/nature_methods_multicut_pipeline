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

    print "Computing lifted features from multople segmentations from %i segmentations for reference segmentation %i" % (ds.n_seg, referenceSegId)

    pLocals = []
    indications = []
    rags = []
    for candidateSegId in xrange(ds.n_seg):
        pLocals.append( learn_and_predict_rf_from_gt(pipelineParam.rf_cache_folder,
            trainsets, ds,
            candidateSegId, candidateSegId,
            featureListLocal, pipelineParam,
            use_2rfs = pipelineParam.use_2rfs) )
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
        energies = numpy.log( (1. - probs) / probs ) + numpy.log( (1. - beta) / beta )

        edge_areas = rag.edgeLengths()

        # weight the energies
        if pipelineParam.weighting_scheme == "z":
            print "Weighting Z edges"
            # z - edges are indicated with 0 !
            area_z_max = float( numpy.max( edge_areas[edge_indications == 0] ) )
            # we only weight the z edges !
            w = weight * edge_areas[edge_indications == 0] / area_z_max
            energies[edge_indications == 0] = numpy.multiply(w, energies[edge_indications == 0])

        elif pipelineParam.weighting_scheme == "xyz":
            print "Weighting xyz edges"
            # z - edges are indicated with 0 !
            area_z_max = float( numpy.max( edge_areas[edge_indications == 0] ) )
            len_xy_max = float( numpy.max( edge_areas[edge_indications == 1] ) )
            # weight the z edges !
            w_z = weight * edge_areas[edge_indications == 0] / area_z_max
            energies[edge_indications == 0] = numpy.multiply(w_z, energies[edge_indications == 0])
            # weight xy edges
            w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
            energies[edge_indications == 1] = numpy.multiply(w_xy, energies[edge_indications == 1])

        elif pipelineParam.weighting_scheme == "all":
            print "Weighting all edges"
            area_max = float( numpy.max( edge_areas ) )
            w = weight * edge_areas / area_max
            energies = numpy.multiply(w, energies)

        uv_ids = numpy.sort(rag.uvIds(), axis = 1)
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
    energies = numpy.log( (1. - probs) / probs ) + numpy.log( (1. - beta) / beta )

    # weight the energies
    if pipelineParam.weighting_scheme == "z":
        print "Weighting Z edges"
        # z - edges are indicated with 0 !
        area_z_max = float( numpy.max( edge_areas[edge_indications == 0] ) )
        # we only weight the z edges !
        w = weight * edge_areas[edge_indications == 0] / area_z_max
        energies[edge_indications == 0] = numpy.multiply(w, energies[edge_indications == 0])

    elif pipelineParam.weighting_scheme == "xyz":
        print "Weighting xyz edges"
        # z - edges are indicated with 0 !
        area_z_max = float( numpy.max( edge_areas[edge_indications == 0] ) )
        len_xy_max = float( numpy.max( edge_areas[edge_indications == 1] ) )
        # weight the z edges !
        w_z = weight * edge_areas[edge_indications == 0] / area_z_max
        energies[edge_indications == 0] = numpy.multiply(w_z, energies[edge_indications == 0])
        # weight xy edges
        w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
        energies[edge_indications == 1] = numpy.multiply(w_xy, energies[edge_indications == 1])

    elif pipelineParam.weighting_scheme == "all":
        print "Weighting all edges"
        area_max = float( numpy.max( edge_areas ) )
        w = weight * edge_areas / area_max
        energies = numpy.multiply(w, energies)

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
