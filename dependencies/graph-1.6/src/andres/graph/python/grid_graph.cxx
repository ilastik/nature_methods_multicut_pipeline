// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL andres_graph_PyArray_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "andres/graph/grid-graph.hxx"
#include "graph_exporter.hxx"

namespace bp = boost::python;
namespace agraph = andres::graph;


agraph::GridGraph<2> * makeGridGraph2D(const size_t sx, const size_t sy){
#ifndef NO_CXX11
    return new agraph::GridGraph<2>({sx,sy});
#else
    agraph::GridGraph<2>::VertexCoordinate shape;
    shape[0] = sx;
    shape[1] = sy;
    return new agraph::GridGraph<2>(shape);
#endif
}

agraph::GridGraph<3> * makeGridGraph3D(const size_t sx, const size_t sy, const size_t sz){
#ifndef NO_CXX11
    return new agraph::GridGraph<3>({sx,sy,sz});
#else
    agraph::GridGraph<3>::VertexCoordinate shape;
    shape[0] = sx;
    shape[1] = sy;
    shape[2] = sz;
    return new agraph::GridGraph<3>(shape);
#endif
}


void exportGridGraph2d(){

   bp::class_<agraph::GridGraph<2> > ("GridGraph2D",bp::init<>())
        .def(GraphExporter<agraph::GridGraph<2> >("GridGraph2D"))
        .def("__init__",bp::make_constructor(&makeGridGraph2D))
   ;
}

void exportGridGraph3d(){

   bp::class_<agraph::GridGraph<3> > ("GridGraph3D",bp::init<>())
        .def(GraphExporter<agraph::GridGraph<3> >("GridGraph3D"))
        .def("__init__",bp::make_constructor(&makeGridGraph3D))
   ;
}
