#ifndef ANDRES_GRAPH_PYTHON_LIFTED_MC_SOLVER_EXPORTER_HXX
#define ANDRES_GRAPH_PYTHON_LIFTED_MC_SOLVER_EXPORTER_HXX

#include <boost/python/def_visitor.hpp>

namespace bp = boost::python;
namespace agraph = andres::graph;



template<class G>
struct GraphName{
    static std::string name(){
        return "Graph";
    }
};

template<>
struct GraphName<agraph::GridGraph<2> >{
    static std::string name(){
        return "GridGraph2D";
    }
};

template<>
struct GraphName<agraph::GridGraph<3> >{
    static std::string name(){
        return "GridGraph3D";
    }
};




template<class SOLVER>
class LiftedMcSolverExporter : 
    public bp::def_visitor<LiftedMcSolverExporter<SOLVER> >
{
public:
    typedef LiftedMcSolverExporter<SOLVER> Exporter;
    typedef SOLVER Self;
    friend class def_visitor_access;

    LiftedMcSolverExporter(
        const std::string & clsName
    )
    :   clsName_(clsName)
    {

    }

    template <class classT>
    void visit(classT& c) const
    {
        c
            .def("run", vigra::registerConverters(&pyRun),
                (
                    bp::arg("labelsIn") = bp::object(),
                    bp::arg("out") = bp::object()
                )
            )
        ;
    }

    // static bp::tuple findEdge(Self & self, const std::size_t u, const std::size_t v){
    //     const auto t = self.findEdge(u,v);
    //     return bp::make_tuple(t.first,t.second);
    // }

    static vigra::NumpyAnyArray pyRun(
        Self & solver, 
        vigra::NumpyArray<1, uint8_t> labelsIn,
        vigra::NumpyArray<1, uint8_t> labelsOut
    ){
        vigra::TinyVector<int, 1> shape(solver.getModel().liftedGraph().numberOfEdges());
        labelsIn.reshapeIfEmpty(shape);
        labelsOut.reshapeIfEmpty(shape);
        vigra::MultiArrayView<1, uint8_t> labelsIn_(labelsIn);
        vigra::MultiArrayView<1, uint8_t> labelsOut_(labelsOut);
        solver.run(labelsIn_, labelsOut_);
        return labelsOut;
    }


    std::string clsName_;
    //std::string factoryName_;
};

#endif
