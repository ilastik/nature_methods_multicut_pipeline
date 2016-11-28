// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL andres_graph_PyArray_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include <vector>


#include "andres/graph/grid-graph.hxx"
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut-lifted/lifted_mc_model.hxx"
#include "andres/graph/multicut-lifted/greedy-additive.hxx"


#include "lifted_mc_solver_exporter.hxx"

namespace bp = boost::python;
namespace agraph = andres::graph;





template<class MODEL, class SETTINGS>
SETTINGS * gaSettingsFactory(
    const MODEL & model
){
    return new SETTINGS();
}

template<class MODEL, class SETTINGS, class SOLVER>
SOLVER * gaSolverFactory(
    const MODEL & model,
    const SETTINGS & settings
){
    return new SOLVER(model, settings);
}



template<class GRAPH>
void exportLiftedGaT(const std::string graphName){

    typedef agraph::multicut_lifted::LiftedMcModel<GRAPH, float> LiftedModel;
    typedef agraph::multicut_lifted::GreedyAdditiveEdgeContraction<LiftedModel> Solver;
    typedef typename Solver::Settings Settings;


    // settings
    const auto settingsClsName = std::string("LiftedGreedyAdditiveSettings") + graphName;
    bp::class_<Settings>(settingsClsName.c_str(),bp::init<>())
        .def_readwrite("nodeStopCond", &Settings::nodeStopCond)
    ;
    // factory
    bp::def("settingsLiftedGreedyAdditive", &gaSettingsFactory<LiftedModel,Settings>, 
            bp::return_value_policy<bp::manage_new_object>());

    // solver
    const auto solverClsName = std::string("LiftedGreedyAdditive") + graphName;
    bp::class_<Solver>(solverClsName.c_str(),bp::no_init)
        .def(LiftedMcSolverExporter<Solver>("LiftedGreedyAdditive"))
    ;
        // factory
    bp::def("liftedGreedyAdditive", &gaSolverFactory<LiftedModel,Settings, Solver>, 
        bp::with_custodian_and_ward_postcall< 0,1 ,
            bp::return_value_policy<   bp::manage_new_object>  >()  
    );
}




void exportLiftedGa(){
    typedef agraph::GridGraph<2> GridGraph2D;
    typedef agraph::GridGraph<3> GridGraph3D;
    typedef agraph::Graph<> Graph;

    exportLiftedGaT<GridGraph2D>("GridGraph2D");
    exportLiftedGaT<GridGraph3D>("GridGraph3D");
    exportLiftedGaT<Graph>("Graph");

}
