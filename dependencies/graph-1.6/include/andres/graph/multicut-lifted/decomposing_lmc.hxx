#pragma once
#ifndef DECOMPOSING_LIFTED_MULTICUT_HXX
#define DECOMPOSING_LIFTED_MULTICUT_HXX

#include <iomanip>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <stack>


#include "andres/partition.hxx"
#include "andres/graph/multicut-lifted/lifted_mc_model.hxx"

namespace andres {
namespace graph {
namespace multicut_lifted {






template<class LIFTED_MC_MODEL, class SOLVER>
class DecomposingLiftedMulticut{
public:

    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef typename SOLVER:: template Rebind<OTHER_LIFTED_MC_MODEL>::type RebindedSolver;
        typedef DecomposingLiftedMulticut<
            OTHER_LIFTED_MC_MODEL, RebindedSolver
        > type;
    };



    typedef LIFTED_MC_MODEL LiftedMcModel;
    typedef typename SOLVER::Settings SolverSettings;

    struct Settings { 

        Settings()
        :   
            decompose(true),
            solverSettings( )
        {

        }
        template<class OTHER>
        Settings(OTHER & other)
        :   
            decompose(other.decompose),
            solverSettings(other.solverSettings)
        {

        }

        bool decompose;
        SolverSettings solverSettings ;

    };


    DecomposingLiftedMulticut(
        const LiftedMcModel & model, 
        const Settings & settings = Settings()
    ) :
        model_(model),
        settings_(settings)
    {

    }

    template<class LABELS_IN, class LABELS_OUT>
    void run(const LABELS_IN & labelsIn, LABELS_OUT & labelsOut){

        if(!settings_.decompose){
            SOLVER solver(model_, settings_.solverSettings);
            solver.run(labelsIn, labelsOut);
        }
        else{
            const auto & originalGraph = model_.originalGraph();
            const auto & liftedGraph = model_.liftedGraph();
            const auto & edgeCosts = model_.edgeCosts();
            Partition<> partition(originalGraph.numberOfVertices());

            // merge all positive nodes
            for(auto e=0; e<liftedGraph.numberOfEdges(); ++e){
                const auto ec = edgeCosts[e];
                if(ec >= 0){
                    const auto v0 = liftedGraph.vertexOfEdge(e, 0);
                    const auto v1 = liftedGraph.vertexOfEdge(e, 1);
                    partition.merge(v0, v1);
                }
            }
            // does problem decompose?
            const auto nrOfProblems = partition.numberOfSets();
            if(nrOfProblems == 1){
                //  the problem does not decompose
                SOLVER solver(model_, settings_.solverSettings);
                solver.run(labelsIn, labelsOut);
            }
            else{
                // the problem does in fact decompose into 
                //  multiple subproblems
                auto nSubProblems = partition.numberOfSets();
                std::map<size_t,size_t> dense;
                partition.representativeLabeling(dense);

                //std::cout<<"number of independent problems "<<partition.numberOfSets()<<"\n";

                // to store data for the individual subproblems
                // (TODO avoid reallocation)
                std::vector< Graph<> * > subOrignalGraphs(nSubProblems);
                std::vector< Graph<> * > subLiftedGraph(nSubProblems);
                std::vector< std::vector<float> > subEdgeCosts(nSubProblems);
                std::vector< std::vector<uint8_t> > subLabelsIn(nSubProblems);
                std::vector< std::vector<uint8_t> > subLabelsOut(nSubProblems);
                std::vector<size_t> nodeCount(nSubProblems, 0);
                std::vector<size_t> nodeToSubGraphNode(originalGraph.numberOfVertices());

                for(auto n=0; n<originalGraph.numberOfVertices(); ++n){
                    const auto denseId = dense[partition.find(n)];
                    nodeToSubGraphNode[n] = nodeCount[denseId];
                    ++nodeCount[denseId];
                }

                // allocate graphs
                for(size_t i=0; i<nSubProblems; ++i){
                    subOrignalGraphs[i] = new Graph<>(nodeCount[i]);
                    subLiftedGraph[i] = new Graph<>(nodeCount[i]);
                }

                // setup sub graphs, sub lifted graphs
                // and sub edge costs for all of the sub problems
                for(auto e=0; e<liftedGraph.numberOfEdges(); ++e){
                    const auto u = liftedGraph.vertexOfEdge(e, 0);
                    const auto v = liftedGraph.vertexOfEdge(e, 1);
                    auto lu = partition.find(u);
                    auto lv = partition.find(v);
                    if(lu!=lv){
                        labelsOut[e] = 1;
                    }
                    else{
                        const auto subProblemIndex = dense[lu];
                        const auto hasEdgeInOrginalGraph = originalGraph.findEdge(u,v).first;
                        auto & subOg = *(subOrignalGraphs[subProblemIndex]);
                        auto & subLg = *(subLiftedGraph[subProblemIndex]);
                        auto & subEc = subEdgeCosts[subProblemIndex];
                        auto & lIn = subLabelsIn[subProblemIndex];
                        const auto subU = nodeToSubGraphNode[u];
                        const auto subV = nodeToSubGraphNode[v];
                        if(hasEdgeInOrginalGraph){
                            subOg.insertEdge(subU,subV);
                        }
                        subLg.insertEdge(subU,subV);
                        subEc.push_back(edgeCosts[e]);
                        lIn.push_back(labelsIn[e]);

                    }
                }
                

                // solve subproblems
                for(auto subProblemIndex=0; subProblemIndex<nSubProblems; ++subProblemIndex){
                    
                    typedef  LiftedMcModelView< Graph<>, Graph<>, std::vector<float>  >  SubModel;
                    typedef typename SOLVER:: template Rebind<SubModel>::type SubSolver;
                    typedef typename SubSolver::Settings SubSolverSettings;
                    auto & subOg = *(subOrignalGraphs[subProblemIndex]);
                    auto & subLg = *(subLiftedGraph[subProblemIndex]);
                    auto & subEc = subEdgeCosts[subProblemIndex];
                    auto & lIn = subLabelsIn[subProblemIndex];
                    auto & lOut = subLabelsOut[subProblemIndex];
                    lOut.resize(lIn.size());
                    SubModel subModel(subOg, subLg,subEc);

                    SubSolverSettings subSolverSettings(settings_.solverSettings);
                    SubSolver subSolver(subModel, subSolverSettings);
                    subSolver.run(lIn, lOut);
                }

                // map subproblem solutions back
                for(auto e=0; e<liftedGraph.numberOfEdges(); ++e){
                    const auto u = liftedGraph.vertexOfEdge(e, 0);
                    const auto v = liftedGraph.vertexOfEdge(e, 1);
                    const auto lu = partition.find(u);
                    const auto lv = partition.find(v);
                    if(lu==lv){
                        const auto subProblemIndex = dense[lu];
                        auto & subLg = *(subLiftedGraph[subProblemIndex]);
                        const auto & lOut = subLabelsOut[subProblemIndex];
                        const auto subU = nodeToSubGraphNode[u];
                        const auto subV = nodeToSubGraphNode[v];
                        labelsOut[e] = lOut[subLg.findEdge(subU,subV).second];
                    }
                }

                // cleanup
                for(size_t i=0; i<nSubProblems; ++i){
                    delete subOrignalGraphs[i];
                    delete subLiftedGraph[i];
                }
            }
        }


    }

private:
    const LiftedMcModel & model_;
    Settings settings_;
};





}
}
}

#endif
