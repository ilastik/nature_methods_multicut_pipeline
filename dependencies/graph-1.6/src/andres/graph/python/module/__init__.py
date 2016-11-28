from _graph import *
import numpy



def _injectorClass(clsToExtend):
    class InjectorClass(object):
        class __metaclass__(clsToExtend.__class__):
            def __init__(self, name, bases, dict):
                for b in bases:
                    if type(b) not in (self, type):
                        for k,v in dict.items():
                            setattr(b,k,v)
                tmp = type.__init__(self, name, bases, dict)
    return InjectorClass



_LiftedMcModelClasses = [
    LiftedMcModelGridGraph2D,LiftedMcModelGridGraph3D,LiftedMcModelGraph
]
for objCls in _LiftedMcModelClasses:


    class _MoreLiftedMcModel(_injectorClass(objCls),objCls):
        
        def setCosts(self, uv, costs, overwrite = True):
            _uv = numpy.require(uv, dtype='uint64')
            _costs = numpy.require(costs, dtype='float32')
            self._setCosts(_uv, _costs, bool(overwrite))

        def setCost(self, u,v, w,overwrite=True):
            self._setCost(int(u),int(v),float(w),bool(overwrite))




def gridGraph(shape):
    if len(shape) == 2:
        return GridGraph2D(int(shape[0]), int(shape[1]))
    elif len(shape) == 3:
        return GridGraph3D(int(shape[0]), int(shape[1]), int(shape[2]))
    else:
        raise RuntimeError("shape has wrong length, GridGraph is only exported to python for 2D and 3D grids")

def liftedMcModel(graph):
    if isinstance(graph, GridGraph2D):
        return LiftedMcModelGridGraph2D(graph)
    elif isinstance(graph, GridGraph3D):
        return LiftedMcModelGridGraph3D(graph)
    elif isinstance(graph, Graph):
        return LiftedMcModelGraph(graph)
    else:
        raise RuntimeError("graph has wrong type")


