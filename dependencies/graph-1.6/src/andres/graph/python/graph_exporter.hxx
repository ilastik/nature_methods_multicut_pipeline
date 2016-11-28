#ifndef ANDRES_GRAPH_PYTHON_GRAPH_EXPORTER_HXX
#define ANDRES_GRAPH_PYTHON_GRAPH_EXPORTER_HXX

#include <boost/python/def_visitor.hpp>

namespace bp = boost::python;
namespace agraph = andres::graph;

template<class GRAPH>
class GraphExporter : public bp::def_visitor<GraphExporter<GRAPH> >
{
public:
    typedef GraphExporter<GRAPH> Exporter;
    typedef GRAPH Self;
    friend class def_visitor_access;

    GraphExporter(const std::string & clsName)
    : clsName_(clsName){
    }

    template <class classT>
    void visit(classT& c) const
    {
        c
            .add_property("numberOfVertices", &Self::numberOfVertices)
            .add_property("numberOfEdges",    &Self::numberOfEdges)

            .def("findEdge",&Exporter::findEdge)
            .def("uvIds",&Exporter::uvIds)
        ;
    }

    static bp::tuple findEdge(Self & self, const std::size_t u, const std::size_t v){
        const auto t = self.findEdge(u,v);
        return bp::make_tuple(t.first,t.second);
    }


    static vigra::NumpyAnyArray uvIds(const Self & self){
        vigra::TinyVector<int, 1> shape(self.numberOfEdges());
        vigra::NumpyArray<1, vigra::TinyVector<uint64_t,2> > out;
        out.reshapeIfEmpty(shape);
        for(auto e=0; e<self.numberOfEdges(); ++e){
            out(e)[0] =  self.vertexOfEdge(e, 0);
            out(e)[1] =  self.vertexOfEdge(e, 1);
        }
        return out;
    }

    std::string clsName_;
};

#endif
