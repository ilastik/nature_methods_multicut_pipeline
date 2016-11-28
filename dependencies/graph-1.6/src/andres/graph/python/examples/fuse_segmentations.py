import graph as agraph
import vigra
import scipy.io as sio
import numpy
from matplotlib import pyplot as plt

def matToGt(mat):
    gts = []
    matContetnt = mat['groundTruth']
    for gt in range(matContetnt.shape[1]):
        gt =  matContetnt[0,gt]['Segmentation']
        gt =  vigra.taggedView(gt[0,0].T,'xy')  
        gts.append(gt)
    return gts

def gtToRagGt(rag, gts):
    ragGts = []
    for gt in gts:
        ragGt,q = rag.projectBaseGraphGt(gt.astype('uint32'))
        ragGt = ragGt.astype('uint32')
        ragGts.append(ragGt[:,None])
    ragGts = numpy.concatenate(ragGts,axis=1)
    return ragGts

folder = "train"
fileNumber = "12003" 

gtFile = "/home/tbeier/datasets/BSR/BSDS500/data/groundTruth/%s/%s.mat"%(folder,fileNumber)
imageFile = "/home/tbeier/datasets/BSR/BSDS500/data/images/%s/%s.jpg"%(folder,fileNumber)

imgRgb = vigra.readImage(imageFile)
shape = imgRgb.shape[0:2]
sp,nSeg = vigra.analysis.slicSuperpixels(imgRgb, intensityScaling=50.0, seedDistance=2)
sp-=1
gg = vigra.graphs.gridGraph(shape)
rag = vigra.graphs.regionAdjacencyGraph(gg, sp)

agraphGG = agraph.gridGraph(shape)

print rag.nodeNum, rag.nodeNum**2


gts = matToGt(sio.loadmat(gtFile))
ragGts = gtToRagGt(rag, gts).astype('uint64')

# convert vigra graph to agraph
assert rag.nodeNum == rag.maxNodeId + 1
graph = agraph.Graph(rag.nodeNum)
uv = rag.uvIds().astype('uint64')
graph.insertEdges(uv)

# get lifted model
model = agraph.liftedMcModel(graph)
# fill objective
agraph.fuseGtObjective(model, ragGts,rr=2,beta=0.5)




if True:
    for i in range(ragGts.shape[1]):
        rag.show(imgRgb,ragGts[:,i].astype('uint32'))
        vigra.show()




# solve the damn thing
settings = agraph.settingsParallelLiftedMc(model)
solver = agraph.parallelLiftedMc(model, settings)
out = solver.run()
nodeLabels = model.edgeLabelsToNodeLabels(out)
nodeLabels = nodeLabels.astype('uint32')

plt.figure()

rag.show(imgRgb, nodeLabels)
vigra.show()

plt.figure()
