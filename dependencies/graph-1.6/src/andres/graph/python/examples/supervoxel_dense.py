import vigra
import numpy
import graph as agraph


def allPairsOfShortestPath(graph,weights):
    pathFinder = vigra.graphs.ShortestPathPathDijkstra(graph)    
    distances = numpy.zeros([graph.nodeNum]*2)
    for ni,node in enumerate(graph.nodeIter()):
        pathFinder.run(weights**1,node)
        d = pathFinder.distances()**1
        distances[ni,:] = d[:]

        ggf  = graph.projectNodeFeaturesToGridGraph(d)
        ggf = vigra.taggedView(ggf,'xy')
        
        if ni<1:
            vigra.imshow(ggf)
            vigra.show()
    print distances.min(),distances.max()
    return distances

show = False
verbose = 1

img = vigra.readImage('12003.jpg')
shape = img.shape[0:2]

# get a rag
sp,nSeg = vigra.analysis.slicSuperpixels(img, intensityScaling=100.0, seedDistance=20)
sp-=1
gg = vigra.graphs.gridGraph(shape)
rag = vigra.graphs.regionAdjacencyGraph(gg, sp)

# get edge weights
gradMag = vigra.filters.structureTensorEigenvalues(img ,1.0  , 1.0)[:,:,0]
gradMag -=gradMag.min()
gradMag /=gradMag.max()
gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromImage(gg,gradMag)

#svigra.imshow(gradMag)
# get region adjacency graph from super-pixel labels
rag = vigra.graphs.regionAdjacencyGraph(gg, sp)
edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)
# accumulate edge weights from gradient magnitude


vigra.segShow(img,sp)
vigra.show()


allPairsOfShortestPath(rag,weights=edgeWeights)
    
