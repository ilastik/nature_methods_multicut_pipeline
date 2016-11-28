// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL andres_graph_PyArray_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>

#include "andres/graph/graph.hxx"
#include "andres/graph/grid-graph.hxx"

#include "graph_exporter.hxx"

namespace bp = boost::python;
namespace agraph = andres::graph;


void insertGridGraphEdges2D(agraph::Graph< > & self, const agraph::GridGraph<2> & gg){
    for(std::size_t e=0; e<gg.numberOfEdges(); ++e){
        auto v0 = gg.vertexOfEdge(e, 0);
        auto v1 = gg.vertexOfEdge(e, 1);
        self.insertEdge(v0, v1);
    }
}
template<class T>
void insertEdges(agraph::Graph< > & self, const vigra::NumpyArray<1, vigra::TinyVector<T, 2> >  & edges){
    for(const auto & edge : edges){
        self.insertEdge(edge[0],edge[1]);
    }
}

void exportGraph(){
    // initialize numpy and vigranumpy
   bp::class_<agraph::Graph< > > ("Graph",bp::init<const size_t >(bp::arg("numberOfNodes")=0))
        .def(GraphExporter<agraph::Graph<> >("Graph"))
        .def("insertGridGraphEdges",&insertGridGraphEdges2D)
        .def("insertEdges",vigra::registerConverters(&insertEdges<int64_t>))
        .def("insertEdges",vigra::registerConverters(&insertEdges<uint64_t>))
        .def("insertEdges",vigra::registerConverters(&insertEdges<int32_t>))
        .def("insertEdges",vigra::registerConverters(&insertEdges<uint32_t>))
   ;
}

