import vigra
import graph as agraph

show = False
verbose = 1

img = vigra.readImage('12074.jpg')#[120:300,0:100,:]
shape = img.shape[0:2]
shape = [s/2 for s in shape]

timg = vigra.sampling.resize(img,shape)

# get orientation

tensor = vigra.filters.structureTensor(timg, 1.0 ,2.0)
eigenrep = vigra.filters.tensorEigenRepresentation2D(tensor)


# print get oriented repulsive edges
edgePmap = eigenrep[:,:,0].copy()
edgePmap -= edgePmap.min()
edgePmap /= edgePmap.max()




graph = agraph.gridGraph(shape)
model = agraph.liftedMcModel(graph)

with vigra.Timer("add long range edges"):
    agraph.addLongRangeEdges(model, edgePmap,0.5,2,5)



settings = agraph.settingsParallelLiftedMc(model)
solver = agraph.parallelLiftedMc(model, settings)

out = solver.run()


nodeLabels = model.edgeLabelsToNodeLabels(out)
nodeLabels = nodeLabels.reshape(shape)

vigra.imshow(nodeLabels)
vigra.show()
