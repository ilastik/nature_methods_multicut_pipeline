import vigra
import graph as agraph

show = False
verbose = 1

img = vigra.readImage('12074.jpg')[120:300,0:100,:]
shape = img.shape[0:2]
graph, liftedGraph = agraph.liftedGridGraph(shape=shape)

# get primitive edge indicator
localWeightsImage = vigra.filters.structureTensorEigenvalues(img ,0.5  , 1.0)[:,:,0]*-1.0
localWeightsImage-=localWeightsImage.min()
localWeightsImage/=localWeightsImage.max()
localWeightsImage-=0.7
if show:
    vigra.imshow(localWeightsImage)

localWeights = agraph.imageToLocalWeights(graph, liftedGraph, localWeightsImage)
if verbose:
    print "localWeightsImage min max", localWeightsImage.min(),localWeightsImage.max()
    print "localWeights min max", localWeights.min(),localWeights.max()


# run lifted fusion moves
result = agraph.liftedMcFusionMoves(graph=graph,
                                    liftedGraph=liftedGraph,
                                    weights=localWeights,
                                    maxNumberOfIterations=2,
                                    maxNumberOfIterationsWithoutImprovement=10,
                                    nodeLimit=40,
                                    verbose=verbose)
if verbose:
    print result.sum()

resultImage = agraph.liftedEdgeLabesToImage(graph=graph,
                                            liftedGraph=liftedGraph,
                                            edgeLabels=result)
resultImage = vigra.taggedView(resultImage,'xy')
vigra.imshow(resultImage)
