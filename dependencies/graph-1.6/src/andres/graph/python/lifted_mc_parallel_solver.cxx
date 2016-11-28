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
#include "andres/graph/multicut-lifted/parallel_lifted_mc.hxx"

#include "lifted_mc_solver_exporter.hxx"

namespace bp = boost::python;
namespace agraph = andres::graph;


template<class MODEL, class SolverSetting>
SolverSetting * pyConstructorSettings2(const MODEL & model){
    return new SolverSetting();
};


template<class SOLVER, class SOLVER_SETTINGS>
SOLVER_SETTINGS * pyConstructorSettings(const typename SOLVER::ProposalGenSettings & pgs){
    auto ss =  new SOLVER_SETTINGS();
    ss->proposalsGenSettings = pgs;
    return ss;
};

template<class MODEL, class SETTINGS, class SOLVER>
SOLVER * pyConstructSolver(
    const MODEL & model,
    const SETTINGS & settings
){
    return new SOLVER(model, settings);
}






template<class SETTINGS>
void addPropoal(
    SETTINGS & settings,
    vigra::NumpyArray<1, uint8_t> proposal
){
    std::vector<uint8_t> p(proposal.begin(), proposal.end());
    settings.externalProposals.push_back(p);
}

template<class SOLVER, class F>
void exportedParallelLiftedMcP(
        const std::string clsNameFull, 
        const std::string clsName,
        F && exportProposalSettings
){
    typedef SOLVER Solver;
    typedef typename Solver::LiftedMcModel Model;
    typedef agraph::multicut_lifted::LiftedMcModel<Model, float> LiftedModel;
    typedef typename Solver::Settings SolverSetting;
    typedef typename Solver::ProposalGen ProposalGen;
    typedef typename ProposalGen::Settings ProposalGenSettings;
    

    auto settingsPgClsName = std::string("SettingsProposalGen") + clsNameFull;
    auto pg = bp::class_<ProposalGenSettings>(settingsPgClsName.c_str(), bp::no_init)
        //.add_property("nParallelProposals",&SolverSetting::nParallelProposals)
    ;
    exportProposalSettings(pg);


    auto settingsClsName = std::string("Settings") + clsNameFull;
    bp::class_<SolverSetting>(settingsClsName.c_str(), bp::no_init)
        .def("addPropoal",vigra::registerConverters(&addPropoal<SolverSetting>))
        .def_readwrite("maxNumberOfIterations",&SolverSetting::maxNumberOfIterations)
        .def_readwrite("nParallelProposals",&SolverSetting::nParallelProposals)
        .def_readwrite("reduceIterations",&SolverSetting::reduceIterations)
        .def_readwrite("seed",&SolverSetting::seed)
        .def_readwrite("verbose",&SolverSetting::verbose)
        .def_readwrite("decompose",&SolverSetting::decompose)
        .def_readwrite("nThreads",&SolverSetting::nThreads)
        .def_readwrite("fmKlTimeLimit",&SolverSetting::fmKlTimeLimit)
        .def_readwrite("stopIfNotImproved",&SolverSetting::stopIfNotImproved)
        //.def_readwrite("proposalsGenSettings",&SolverSetting::proposalsGenSettings)
    ;


    bp::class_<Solver>(clsNameFull.c_str(), bp::no_init)
        .def(LiftedMcSolverExporter<Solver>(clsName))
    ;


    auto fSettigsProposalsName = std::string("settingsProposals") + clsName;
    bp::def(fSettigsProposalsName.c_str(), &pyConstructorSettings2<Model, ProposalGenSettings>, 
            bp::return_value_policy<bp::manage_new_object>());

    auto fSettigsName = std::string("settingsFusionMoves");
    bp::def(fSettigsName.c_str(), &pyConstructorSettings<Solver, SolverSetting>, 
            bp::return_value_policy<bp::manage_new_object>());

    auto fname = std::string("fusionMoves");
    fname[0] = std::tolower(fname[0]);
    bp::def(fname.c_str(), &pyConstructSolver<Model, SolverSetting, Solver>, 
        bp::with_custodian_and_ward_postcall< 0,1 ,
            bp::return_value_policy<   bp::manage_new_object>  >()  
    );
}








template<class GRAPH>
void exportedParallelLiftedMcT(const std::string graphName){

    typedef agraph::multicut_lifted::LiftedMcModel<GRAPH, float> LiftedModel;

    {
        auto baseName = std::string("FusionMovesRandomizedProposals");
        typedef agraph::multicut_lifted::RandomizedProposalGenerator<LiftedModel> ProposalGen;
        typedef agraph::multicut_lifted::ParallelSolver<LiftedModel,ProposalGen> Solver;
        typedef typename ProposalGen::Settings SettingsPg;
        auto f = [](bp::class_<SettingsPg> & cls){
            cls
                .def_readwrite("sigma", &SettingsPg::sigma)
                .def_readwrite("nodeLimit", &SettingsPg::nodeLimit)
                //.def_readwrite("seed", &SettingsPg::seed)
                .def_readwrite("useGA", &SettingsPg::useGA)
            ;
        };
        exportedParallelLiftedMcP<Solver>(baseName+graphName, baseName, f);
    }
    {
        auto baseName = std::string("FusionMovesSubgraphProposals");
        typedef agraph::multicut_lifted::SubgraphProposalGenerator<LiftedModel> ProposalGen;
        typedef agraph::multicut_lifted::ParallelSolver<LiftedModel,ProposalGen> Solver;
        typedef typename ProposalGen::Settings SettingsPg;
        auto f = [](bp::class_<SettingsPg> & cls){
            cls
                .def_readwrite("subgraphRadius", &SettingsPg::subgraphRadius)
                //.def_readwrite("seed", &SettingsPg::seed)
            ;
        };
        exportedParallelLiftedMcP<Solver>(baseName+graphName, baseName, f);
    }
    {
        auto baseName = std::string("FusionMovesRescaledProposals");
        typedef agraph::multicut_lifted::RescaledProposalGenerator<LiftedModel> ProposalGen;
        typedef agraph::multicut_lifted::ParallelSolver<LiftedModel,ProposalGen> Solver;
        typedef typename ProposalGen::Settings SettingsPg;
        auto f = [](bp::class_<SettingsPg> & cls){
            cls
                .def_readwrite("reducingFactorMean", &SettingsPg::reducingFactorMean)
                .def_readwrite("reducingFactorSigma", &SettingsPg::reducingFactorSigma)
                //.def_readwrite("seed", &SettingsPg::seed)
            ;
        };
        exportedParallelLiftedMcP<Solver>(baseName+graphName, baseName, f);
    }
}




void exportedParallelLiftedMc(){
    typedef agraph::GridGraph<2> GridGraph2D;
    typedef agraph::GridGraph<3> GridGraph3D;
    typedef agraph::Graph<> Graph;

    exportedParallelLiftedMcT<GridGraph2D>("GridGraph2D");
    exportedParallelLiftedMcT<GridGraph3D>("GridGraph3D");
    exportedParallelLiftedMcT<Graph>("Graph");

}
