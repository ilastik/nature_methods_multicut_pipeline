#pragma once
#ifndef ANDRES_GRAPH_LIFTED_MC_BLOCK_ICM_HXX_
#define ANDRES_GRAPH_LIFTED_MC_BLOCK_ICM_HXX_

#include <iomanip>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <stack>
#include <set>

#include "andres/partition.hxx"
#include "andres/graph/multicut-lifted/lifted_mc_model.hxx"
#include "andres/graph/neighborhood.hxx"

namespace andres {
namespace graph {
namespace multicut_lifted {




template<class SET, class T>
inline bool inSet(const SET & set, const T & v){
    return set.find(v) != set.end();
}

template<class SOLVER>
class BlockIcmLiftedMc{
public:
    typedef std::set<size_t> Set;
    typedef typename SOLVER::LiftedMcModel LiftedMcModel;
    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef typename SOLVER:: template Rebind<OTHER_LIFTED_MC_MODEL>::type RebindedSolver;
        typedef BlockIcmLiftedMc<RebindedSolver>  type;
    };

    typedef typename SOLVER::Settings SolverSettings;

    struct Settings { 

        Settings()
        :   
            numberOfIterations(10000),
            subgraphRadius(1),
            seed(-1),
            softInf(1000000.0),
            verbose(0),
            solverSettings( )
        {

        }
        template<class OTHER>
        Settings(OTHER & other)
        :   
            numberOfIterations(other.numberOfIterations),
            subgraphRadius(other.subgraphRadius),
            seed(other.seed),
            verbose(other.verbose),
            solverSettings(other.solverSettings)
        {

        }

        int numberOfIterations;
        int subgraphRadius;
        int seed;
        float softInf;
        int verbose;
        SolverSettings solverSettings ;

    };


    struct BoundaryConditionMask2 {

        BoundaryConditionMask2(
            const LiftedMcModel & model,
            const Set &     core,
            const Set &     localBorder,
            const std::vector<size_t> & edgesInLiftedGraph,
            const std::vector<uint8_t> & currentEdgeLabels
        )
        :   
        model_(model),
        core_(core),
        localBorder_(localBorder),
        edgesInLiftedGraph_(edgesInLiftedGraph),
        currentEdgeLabels_(currentEdgeLabels),
        startVertex_()
        {

        }

        typedef std::size_t Value;

        bool vertex(const Value v) const
        {   

            if(inSet(core_, v)){
                // starting vertex needs to be added
                return v==startVertex_;
            }
            else{
                return true;
            }
        }

        bool edge(const Value e) const
        {   
            const auto eInLifted = edgesInLiftedGraph_[e];
            if(currentEdgeLabels_[eInLifted]==1){
                return false;
            }
            else{
                const auto u = model_.liftedGraph().vertexOfEdge(eInLifted, 0);
                const auto v = model_.liftedGraph().vertexOfEdge(eInLifted, 1);
                // case 1 : both in inner subgraph  => false
                if(inSet(core_,u) && inSet(core_,v)){
                    return false;
                }
                else{
                    return true;
                }
            }
        }
        const LiftedMcModel & model_;
        const Set & core_;
        const Set & localBorder_;
        const std::vector<size_t > & edgesInLiftedGraph_;
        const std::vector<uint8_t> & currentEdgeLabels_;
        size_t startVertex_;
    };
    BlockIcmLiftedMc(
        const LiftedMcModel & model, 
        const Settings & settings = Settings()
    ) :
        model_(model),
        settings_(settings)
    {
                
    }

    template<class LABELS_IN, class LABELS_OUT>
    void run(const LABELS_IN & labelsIn, LABELS_OUT & labelsOut){

        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts  =model_.edgeCosts();



        // some lambdas
        
        // lambda to compute energy 
        auto getEnergy = [&] (const std::vector<uint8_t> & edgeLabels_) {
            auto totalCost = 0.0;
            for(std::size_t edge = 0; edge < liftedGraph.numberOfEdges(); ++edge){
                if(edgeLabels_[edge]){
                    totalCost += edgeCosts[edge];
                }
            }
            return totalCost;
        };

        // random var
        std::default_random_engine generator;
        std::uniform_int_distribution<> uintDist(0,originalGraph.numberOfVertices()-1);
        auto getCenterVertex = [&](){ return uintDist(generator); };

        // grow subgraph
        auto getSubgraph = [&](const size_t centerVertex){
            return std::move(verticesInGraphNeigborhood(model_.originalGraph(),
                centerVertex, settings_.subgraphRadius));
        };

        auto getLocalBorder = [&](const Set & core){
            Set ret;
            for(const auto node : core){
                for(auto aIter = originalGraph.adjacenciesFromVertexBegin(node); aIter != originalGraph.adjacenciesFromVertexEnd(node); ++aIter){
                    const auto a = *aIter;
                    const auto otherNode  = a.vertex();
                    if(core.find(otherNode)==core.end()){
                        ret.insert(otherNode);
                    }
                }
            }
            return std::move(ret);
        };

        auto getLiftedBorder = [&](const Set & core, const Set & localBorder){
            Set ret;
            for(const auto node : core){
                for(auto aIter = liftedGraph.adjacenciesFromVertexBegin(node); aIter != liftedGraph.adjacenciesFromVertexEnd(node); ++aIter){
                    const auto a = *aIter;
                    const auto otherNode  = a.vertex();
                    if(!inSet(core, otherNode) && !inSet(localBorder, otherNode)){
                       ret.insert(otherNode);
                    }
                }
            }
            return std::move(ret);
        };

        std::vector<std::size_t> edgesInLiftedGraph(originalGraph.numberOfEdges());
        for (std::size_t i = 0; i < originalGraph.numberOfEdges(); ++i){
            auto v0 = originalGraph.vertexOfEdge(i, 0);
            auto v1 = originalGraph.vertexOfEdge(i, 1);
            edgesInLiftedGraph[i] = liftedGraph.findEdge(v0, v1).second;
        }



        std::vector<uint8_t>  currentEdgeLabels(liftedGraph.numberOfEdges(), 1);
        std::copy(labelsIn.begin(), labelsIn.end(), currentEdgeLabels.begin());
        std::vector<uint64_t> graphToSubgraph(model_.originalGraph().numberOfVertices());
        // outer iterations


        auto printer = [&](
            const std::string & name,
            std::function<void(size_t)> f
        ){
            std::cout<<name<<":\n";
            for(auto yy=0; yy<originalGraph.shape(1); ++yy){
            for(auto xx=0; xx<originalGraph.shape(0); ++xx){
                    //std::cout.width(3);
                    auto var = xx+yy*originalGraph.shape(0);
                    f(var);
                }
                std::cout<<"\n";
            }
        };





        for(auto i=0; i<settings_.numberOfIterations; ++i){


            auto ccLabels = getNodeLabels(currentEdgeLabels, edgesInLiftedGraph);


            // get a center variable
            const auto centerVertex = getCenterVertex();

            // get the core
            auto core  = getSubgraph(centerVertex);

            //  get the local border
            auto localBorder = getLocalBorder(core);

            // get the lifted border
            auto liftedBorder = getLiftedBorder(core, localBorder);

            auto totalBorder = localBorder;
            totalBorder.insert(liftedBorder.begin(), liftedBorder.end());


            // merge all together into the complete subgraph
            Set subgraph  = core;
            subgraph.insert(localBorder.begin(), localBorder.end());
            subgraph.insert(liftedBorder.begin(), liftedBorder.end());
            




            if(false){//settings_.verbose >=2){
                printer("cc",[&](size_t var){
                    std::cout.width(3);
                    std::cout<<ccLabels[var]<<"  ";
                });

                printer("borders",[&](size_t var){
                    std::cout.width(3);
                    if(inSet(subgraph,var))
                        if(inSet(localBorder,var))
                            std::cout<<"2  ";
                        else if(inSet(liftedBorder,var))
                            std::cout<<"3  ";
                        else
                            std::cout<<"1  ";
                    else
                        std::cout<<"0  ";
                });
            }  




            // local to global mapping / global to local 
            std::vector<uint64_t> subGraphToGraph(subgraph.size());
            auto subVertex = 0;
            for(auto vertex : subgraph){
                subGraphToGraph[subVertex] = vertex;
                graphToSubgraph[vertex]  = subVertex;
                ++subVertex;
            }
            
            // set up the sub graphs
            Graph<> subOriginalGraph(subgraph.size());
            Graph<> subLiftedGraph(subgraph.size());
            std::vector<float>      subEdgeCosts;
            std::vector<uint8_t>    subLabelsIn;




            std::map<size_t,uint8_t> subgraphFixedEdges;

            auto cCost = [&](uint8_t state){
                return state == 0 ? settings_.softInf : -1.0*settings_.softInf;
            };

            auto addCoreCoreEdge = [&] (
                const size_t u, const size_t v,
                const size_t subU, const size_t subV,
                const size_t edge,
                const bool isLocalEdge
            ){
                GRAPH_CHECK_OP(subU,!=,subV,"");
                if(isLocalEdge){
                    subOriginalGraph.insertEdge(subU,subV);
                }
                subLiftedGraph.insertEdge(subU,subV);
                subEdgeCosts.push_back(edgeCosts[edge]);
                subLabelsIn.push_back(currentEdgeLabels[edge]);
            };

            auto addCoreToAnyBorderEdge= [&] (
                const size_t u, const size_t v,
                const size_t subU, const size_t subV,
                const size_t edge,
                const bool isLocalEdge
            ){
                GRAPH_CHECK_OP(subU,!=,subV,"");
                auto c = edgeCosts[edge];
                auto s = currentEdgeLabels[edge];

                if(isLocalEdge){
                    subOriginalGraph.insertEdge(subU,subV);
                    c = cCost(s);
                }
                
                auto eSub = subLiftedGraph.insertEdge(subU,subV);
                if(isLocalEdge)
                    subgraphFixedEdges[eSub] = s;
                subEdgeCosts.push_back(c);
                subLabelsIn.push_back(s);
                
            };

            auto addAnyBorderToAnyBorderEdge= [&] (
                const size_t u, const size_t v,
                const size_t subU, const size_t subV,
                const size_t edge,
                const bool isLocalEdge
            ){
                GRAPH_CHECK_OP(subU,!=,subV,"");
                auto c = edgeCosts[edge];
                auto s = currentEdgeLabels[edge];
                if(isLocalEdge){
                    subOriginalGraph.insertEdge(subU,subV);
                    c = cCost(s);
                    
                    auto eSub = subLiftedGraph.insertEdge(subU,subV);
                    subgraphFixedEdges[eSub] = s;
                    subEdgeCosts.push_back(c);
                    subLabelsIn.push_back(currentEdgeLabels[edge]);
                }
                else{
                    auto eSub = subLiftedGraph.insertEdge(subU,subV);
                    //subgraphFixedEdges[eSub] = s;
                    subEdgeCosts.push_back(c);
                    subLabelsIn.push_back(currentEdgeLabels[edge]);
                }
                
            };


            // build the subgraphs and sub edgeCosts
            Set usedLfitedEdges;

            for(auto u : subgraph){

                const auto subU = graphToSubgraph[u];
                // iterate over all lifted edges
                for(auto aIter = liftedGraph.adjacenciesFromVertexBegin(u); aIter != liftedGraph.adjacenciesFromVertexEnd(u); ++aIter){

                    const auto a = *aIter;
                    const auto edge = a.edge();
                    const auto v  = a.vertex();

                    

                    if(inSet(subgraph, v) && !inSet(usedLfitedEdges, edge)){

                        const auto subV = graphToSubgraph[v];
                        const bool isLocalEdge = originalGraph.findEdge(u, v).first;

                        usedLfitedEdges.insert(edge);


                        //std::cout<<"add "<<u<<" "<<v<<" local? "<<isLocalEdge<<"\n";
                        //  first node in core
                        if(inSet(core, u)){
                            //std::cout<<u<<" in core ";
                            //  second node in core
                            if(inSet(core, v)){
                                //std::cout<<" "<<v<<" in core\n";
                                addCoreCoreEdge(u,v,subU,subV,edge, isLocalEdge);
                            }
                            // second node in local border
                            else if(inSet(localBorder, v)){
                                //std::cout<<" "<<v<<" in localBorder\n";
                                addCoreToAnyBorderEdge(u,v,subU,subV,edge, isLocalEdge);
                            }
                            // second node in lifted border
                            else{
                                //std::cout<<" "<<v<<" in liftedBorder\n";
                                addCoreToAnyBorderEdge(u,v,subU,subV,edge, isLocalEdge);
                            }
                        }
                        // first node in local border
                        else if(inSet(localBorder, u)){
                            //std::cout<<u<<" in localBorder ";
                            //  second node in core
                            if(inSet(core, v)){
                                 //std::cout<<" "<<v<<" in core\n";
                                addCoreToAnyBorderEdge(v,u,subV,subU,edge, isLocalEdge);
                            }
                            // second node in local border
                            else if(inSet(localBorder, v)){
                                //std::cout<<" "<<v<<" in localBorder\n";
                                addAnyBorderToAnyBorderEdge(u,v,subU,subV ,edge, isLocalEdge);
                            }
                            // second node in lifted border
                            else{
                                 //std::cout<<" "<<v<<" in liftedBorder\n";
                                addAnyBorderToAnyBorderEdge(u,v,subU,subV ,edge, isLocalEdge);
                            }
                        }
                        // first node in lifted border
                        else{
                            //std::cout<<u<<" in liftedBorder ";
                            //  second node in core
                            if(inSet(core, v)){
                                //std::cout<<" "<<v<<" in core\n";
                                addCoreToAnyBorderEdge(v,u,subV,subU,edge, isLocalEdge);
                            }
                            // second node in local border
                            else if(inSet(localBorder, v)){
                                //std::cout<<" "<<v<<" in localBorder\n";
                                addAnyBorderToAnyBorderEdge(u,v,subU,subV ,edge, isLocalEdge);
                            }
                            // second node in lifted border
                            else{
                                //std::cout<<" "<<v<<" in liftedBorder\n";
                                addAnyBorderToAnyBorderEdge(u,v,subU,subV ,edge, isLocalEdge);
                            }
                        }
                    }
                }   
            }

            // add the boundary conditions

            BoundaryConditionMask2 mask(model_, core, localBorder, edgesInLiftedGraph, 
                                       currentEdgeLabels);


            std::map<size_t, uint8_t> lrEdges;
            if(totalBorder.size() >=2){

                for(const auto startVertex : totalBorder){
                    const auto u = startVertex;
                    mask.startVertex_ = startVertex;
                    //std::cout<<"\n\n"<<startVertex<<"\n";
                    Set reachedBorderNodes;
                    // the callback
                    auto callback = [&](const size_t node, const size_t depth, 
                                        bool & proceed,bool & add ){  
                        proceed = true;
                        add = true;
                            
                        if(node != startVertex &&  inSet(totalBorder, node) ){
                            //GRAPH_CHECK(subgraphNodeSet.find(node)!=subgraphNodeSet.end(),"");
                            reachedBorderNodes.insert(node);
                            //add = false;
                            if(reachedBorderNodes.size() == totalBorder.size()-1)
                                proceed = false;
                        }
                    };
                    breadthFirstSearch(originalGraph, mask, startVertex, callback);

                    //std::cout<<"reachedBorderNodes" <<reachedBorderNodes.size()<<"\n";
                    for(const auto v : reachedBorderNodes){
                        if(u<v){

                            
                            auto cost = cCost(0);
               
                            const auto subU = graphToSubgraph[u];
                            const auto subV = graphToSubgraph[v];
                            GRAPH_CHECK_OP(subU,<,subOriginalGraph.numberOfVertices(),"");
                            GRAPH_CHECK_OP(subV,<,subOriginalGraph.numberOfVertices(),"");

                            const auto fe = subLiftedGraph.findEdge(subU,subV);
                            const auto feo = subOriginalGraph.findEdge(subU,subV);
                            if(fe.first){
                                subgraphFixedEdges[fe.second] = 0;
                                subEdgeCosts[fe.second] = cost;
                                subLabelsIn[fe.second] = 0;
                                if(!feo.first)
                                    subOriginalGraph.insertEdge(subU,subV);
                            }
                            else{
                                GRAPH_CHECK(!feo.first,"");
                                subOriginalGraph.insertEdge(subU,subV);
                                auto subE = subLiftedGraph.insertEdge(subU,subV);
                                subgraphFixedEdges[subE] = 0;
                                subEdgeCosts.push_back(cost);
                                subLabelsIn.push_back(0);
                            }
                        }
                    }
                }
            }


            // optimize the submodel
            //std::cout<<"submodel n nodes     "<<subOriginalGraph.numberOfVertices()<<"\n";
            //std::cout<<"core n nodes         "<<core.size()<<"\n";
            //std::cout<<"localBorder n nodes  "<<localBorder.size()<<"\n";
            //std::cout<<"liftedBorder n nodes "<<liftedBorder.size()<<"\n";

            for(auto e : subgraphFixedEdges){
                auto subE = e.first;
                auto s    = e.second;
                const auto subU = subLiftedGraph.vertexOfEdge(subE, 0);
                const auto subV = subLiftedGraph.vertexOfEdge(subE, 1);
                const auto u = subGraphToGraph[subU];
                const auto v = subGraphToGraph[subV];

                //std::cout<<"constraint "<< u << " "<<v<< " to "<<int(s)<<"\n"; 
            }


            std::vector<uint8_t> subLabelsOut(subLabelsIn.size());
            typedef LiftedMcModelView<Graph<>, Graph<>, std::vector<float> > SubModel;
            SubModel subModel(subOriginalGraph, subLiftedGraph, subEdgeCosts);
            solveSubProblem(subModel, subLabelsIn, subLabelsOut);


            bool violated = false;

            //std::cout<<"subgraph fixed edges "<<subgraphFixedEdges.size()<<"\n";
            // violated bc edges
            for(auto e : subgraphFixedEdges){
                if(int(subLabelsOut[e.first]) != int(e.second)){
                    //std::cout<<"sl = "<<int(subLabelsOut[e.first])<< "should "<<int(e.second)<<"\n";
                    violated = true;
                    break;
                }
            }
            if(false){
                std::cout<<"violated\n";
            }
            else{

                auto clBuffer = currentEdgeLabels;
                for(auto subE=0; subE<subLiftedGraph.numberOfEdges(); ++subE){
                    const auto subU = subLiftedGraph.vertexOfEdge(subE, 0);
                    const auto subV = subLiftedGraph.vertexOfEdge(subE, 1);
                    const auto u = subGraphToGraph[subU];
                    const auto v = subGraphToGraph[subV];
                    const auto findE = liftedGraph.findEdge(u,v);
                    if(findE.first){
                        auto newLabel = int(subLabelsOut[subE]); 
                        //if (newLabel != clBuffer[findE.second])
                        //    std::cout<<u<<" "<<v<<"  = "<<int(subLabelsOut[subE])<<" DIFF\n";
                        //else
                        //    std::cout<<u<<" "<<v<<"  = "<<int(subLabelsOut[subE])<<"\n";
                        clBuffer[findE.second] = subLabelsOut[subE];
                    }
                }
                makeValid(clBuffer,edgesInLiftedGraph, clBuffer);
                //std::cout<<"current Energy "<<getEnergy(currentEdgeLabels)<<"\n";
                //std::cout<<"sub     Energy "<<getEnergy(clBuffer)<<"\n";
                auto newEnergy = getEnergy(clBuffer);
                if(newEnergy<getEnergy(currentEdgeLabels)){
                    currentEdgeLabels  = clBuffer;
                    std::cout<<"Energy "<<getEnergy(currentEdgeLabels)<<"\n";
                }
                //if(settings_.verbose >=1)
                
            }
        }
        std::copy(currentEdgeLabels.begin(), currentEdgeLabels.end(), labelsOut.begin());
    }

private:




    template<class SUBMODEL>
    void solveSubProblem(
        const SUBMODEL & subModel,
        std::vector<uint8_t> & subLabelsIn,
        std::vector<uint8_t> & subLabelsOut
    ){


            // lambda to compute energy 
            auto getSubEnergy = [&] (const std::vector<uint8_t> & edgeLabels_) {
                auto totalCost = 0.0;
                for(std::size_t edge = 0; edge < subModel.liftedGraph().numberOfEdges(); ++edge){
                    if(edgeLabels_[edge]){
                        totalCost += subModel.edgeCosts()[edge];
                    }
                }
                return totalCost;
            };


            typedef typename SOLVER:: template Rebind<SUBMODEL>::type SubSolver;
            typedef typename SubSolver::Settings SubSolverSettings;

            SubSolverSettings subSolverSettings(settings_.solverSettings);
            SubSolver subSolver(subModel, subSolverSettings);
            
            GRAPH_CHECK_OP(subLabelsOut.size(), == , subModel.liftedGraph().numberOfEdges(),"");

            // run with 
            std::vector<uint8_t> solL(subLabelsOut.size());
            std::vector<uint8_t> sol0(subLabelsOut.size());
            std::vector<uint8_t> sol1(subLabelsOut.size());

            //std::cout<<"solve ll\n";
            std::fill(subLabelsIn.begin(), subLabelsIn.end(), 1);
            subSolver.run(subLabelsIn, subLabelsOut);
            //subSolver.run(subLabelsOut, subLabelsOut);
            //std::cout<<"solve 1\n";
            //std::fill(subLabelsIn.begin(), subLabelsIn.end(), 0);
            //subSolver.run(subLabelsIn, subLabelsOut);

            //std::cout<<"solve 0\n";
            //std::fill(subLabelsIn.begin(), subLabelsIn.end(), 1);
            //subSolver.run(subLabelsIn, subLabelsOut);

            if(false){

                    const auto eSolL = getSubEnergy(solL);
                    const auto eSol0 = getSubEnergy(sol0);
                    const auto eSol1 = getSubEnergy(sol1);
                    //std::cout<<"E L "<<eSolL<<"\n";
                    //std::cout<<"E 0 "<<eSol0<<"\n";
                    //std::cout<<"E 1 "<<eSol1<<"\n";

                    if(eSolL <= std::min(eSol0,eSol1))
                        subLabelsOut = solL;
                    else if(eSol0 <= std::min(eSolL,eSol1))
                        subLabelsOut = sol0;
                    else
                        subLabelsOut = sol1;
            }

    }

    std::vector<size_t> 
    getNodeLabels(
        const std::vector<uint8_t> & labels, 
        const std::vector<size_t> & edgesInLiftedGraph
    ){
        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts  =model_.edgeCosts();
        struct SubgraphWithCut { // a subgraph with cut mask
            SubgraphWithCut(const std::vector<uint8_t> & labels, std::vector<std::size_t> const& edgesInLiftedGraph)
                : labels_(labels), edgesInLiftedGraph_(edgesInLiftedGraph)
                {}
            bool vertex(const std::size_t v) const
                { return true; }
            bool edge(const std::size_t e) const
                { return labels_[edgesInLiftedGraph_[e]] == 0; }

            std::vector<std::size_t> const& edgesInLiftedGraph_;
            const std::vector<uint8_t> & labels_;
        };

        // build decomposition based on the current multicut
        ComponentsBySearch<typename LiftedMcModel::OriginalGraph> components;
        components.build(originalGraph, SubgraphWithCut(labels, edgesInLiftedGraph));


        // find out how many connected components there are
        // check if the input multicut labeling is valid
        for(std::size_t edge = 0; edge < liftedGraph.numberOfEdges(); ++edge)
        {

            auto v0 = liftedGraph.vertexOfEdge(edge, 0);
            auto v1 = liftedGraph.vertexOfEdge(edge, 1);
            if (static_cast<bool>(labels[edge]) != !components.areConnected(v0, v1)){
                std::cout<<v0<<" "<<v1<<"  label "<<labels[edge]<< "connected? "<<components.areConnected(v0,v1)<<"\n";
                throw std::runtime_error("the current multicut labeling is invalid.");
            }
        }   
        return std::move(components.labels_);
    }

    void
    makeValid(
        const std::vector<uint8_t> & labelsIn, 
        const std::vector<size_t> & edgesInLiftedGraph,
        std::vector<uint8_t> & labelsOut
    ){
        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts  =model_.edgeCosts();
        struct SubgraphWithCut { // a subgraph with cut mask
            SubgraphWithCut(const std::vector<uint8_t> & labels, std::vector<std::size_t> const& edgesInLiftedGraph)
                : labels_(labels), edgesInLiftedGraph_(edgesInLiftedGraph)
                {}
            bool vertex(const std::size_t v) const
                { return true; }
            bool edge(const std::size_t e) const
                { return labels_[edgesInLiftedGraph_[e]] == 0; }

            std::vector<std::size_t> const& edgesInLiftedGraph_;
            const std::vector<uint8_t> & labels_;
        };

        // build decomposition based on the current multicut
        ComponentsBySearch<typename LiftedMcModel::OriginalGraph> components;
        components.build(originalGraph, SubgraphWithCut(labelsIn, edgesInLiftedGraph));


        // find out how many connected components there are
        // check if the input multicut labeling is valid
        for(std::size_t edge = 0; edge < liftedGraph.numberOfEdges(); ++edge)
        {

            auto v0 = liftedGraph.vertexOfEdge(edge, 0);
            auto v1 = liftedGraph.vertexOfEdge(edge, 1);
            labelsOut[edge] = !components.areConnected(v0, v1);
        }   
    }

    const LiftedMcModel & model_;
    Settings settings_;


};





}
}
}

#endif
