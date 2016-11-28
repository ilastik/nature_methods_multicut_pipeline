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
        shape = gt.shape
        newShape = [s/1 for s in shape]
        gtf = gt.astype('float32')

        gtfs = vigra.sampling.resizeImageNoInterpolation(gtf, newShape)
        gt = gtfs.astype('uint32')


        #vigra.imshow(gt)
        #vigra.show()
        gts.append(gt[:,:,None])
    gts = numpy.concatenate(gts,axis=2)
    return gts





class LiftedAMUS(object):
    def __init__(self, gts):
        self.shape = gts.shape[0:2]
        self.nExperts = gts.shape[2]
        self.gts =numpy.array(gts.astype("uint64"))
        self.agraphGG = agraph.gridGraph(self.shape)
        self.pExperts = numpy.ones(shape=[self.nExperts],dtype='float64')/self.nExperts
        self.gtAsCut = None
        print self.pExperts.dtype,"is the experts dtype",self.pExperts.shape



        self._runAll()

    def _runAll(self):
        for runNr in range(20):
            model = self._buildModel()
            averaged = self.optimizeModel(model)
            c = self.updatePExperts(averaged)
            if c<0.00001:
                break

        vigra.segShow(imgRgb,self.lastNodeLabels+1)
        vigra.show()

    def updatePExperts(self,averaged):
        old = self.pExperts.copy()
        for ie, gtc in enumerate(self.gtAsCut):
            av = numpy.average(numpy.abs(averaged.astype('int') - gtc.astype('int'))).astype('float32')
            p = 1.0 - av
            self.pExperts[ie] = p

        self.pExperts /= numpy.sum(self.pExperts)
        print "pExperts",self.pExperts

        self.pExperts = numpy.exp(30.0*self.pExperts)
        self.pExperts /= numpy.sum(self.pExperts)

        convergence = numpy.sum(numpy.abs(old-self.pExperts))

        print "pExperts",self.pExperts
        print "convergence",convergence
        return convergence
    def _buildModel(self):
        # get lifted model
        model = agraph.liftedMcModel(agraphGG)
        # fill objective
        #agraph.fuseGtObjectiveGrid(model, gts,rr=10,beta=0.25)
        #self.
        agraph.fuseGtObjectiveGrid(model,self.gts,self.pExperts, rr=8,beta=0.7, cLocal=500)

        return model

    def optimizeModel(self, model):
        
        if self.gtAsCut is None:
            # generate external proposals
            # ( we should do this only once)
            self.gtAsCut = []
            for gti in range(self.nExperts):
                gt2d = self.gts[:,:,gti].astype('uint64')
                #print gt2d.shape
                gt1d = model.flattenLabels(gt2d)
                #print gt1d
                edgeGt = model.nodeLabelsToEdgeLabels(gt1d)
                self.gtAsCut.append(edgeGt)


        # settings for proposal generator
        settingsProposalGen = agraph.settingsProposalsFusionMovesSubgraphProposals(model)
        settingsProposalGen.subgraphRadius = 20


        # settings for solver itself
        settings = agraph.settingsFusionMoves(settingsProposalGen)
        settings.maxNumberOfIterations = 4
        settings.nParallelProposals = 50
        settings.reduceIterations = 0
        settings.seed = 42
        for ep in self.gtAsCut:
            settings.addPropoal(ep)

        # solver
        solver = agraph.fusionMoves(model, settings)

        # solve the damn thing
        out = solver.run()
        nodeLabels = model.edgeLabelsToNodeLabels(out)
        nodeLabels = nodeLabels.astype('uint32')

        nodeLabels = nodeLabels.reshape(shape,order='F')
        nodeLabels = vigra.taggedView(nodeLabels,'xy')

        self.lastNodeLabels = nodeLabels
        vigra.segShow(imgRgb,nodeLabels+1)
        vigra.show()

        return out







folder = "train"
fileNumber = "66075" 

gtFile = "/home/tbeier/datasets/BSR/BSDS500/data/groundTruth/%s/%s.mat"%(folder,fileNumber)
imageFile = "/home/tbeier/datasets/BSR/BSDS500/data/images/%s/%s.jpg"%(folder,fileNumber)

imgRgb = vigra.readImage(imageFile)
newShape = [s/1 for s in imgRgb.shape[0:2]]
imgRgb = vigra.sampling.resize(imgRgb, newShape)
shape = imgRgb.shape[0:2]
agraphGG = agraph.gridGraph(shape)


gts = matToGt(sio.loadmat(gtFile)).astype('uint64')
LiftedAMUS(gts=gts)





