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

    struct BoundaryConditionMask {

        BoundaryConditionMask(
            const LiftedMcModel & model,
            const Set &     innerNodeSet,
            const Set &     totalNodeSet,
            const Set &     innerBorder,
            const Set &     outerBorder,
            const std::vector<size_t> & edgesInLiftedGraph,
            const std::vector<uint8_t> & currentEdgeLabels
        )
        :   
        model_(model),
        innerNodeSet_(innerNodeSet),
        totalNodeSet_(totalNodeSet),
        innerBorder_(innerBorder),
        outerBorder_(outerBorder),
        edgesInLiftedGraph_(edgesInLiftedGraph),
        currentEdgeLabels_(currentEdgeLabels),
        startVertex_()
        {

        }

        typedef std::size_t Value;

        bool vertex(const Value v) const
        {   
            // if the vertex is not in the inner node set we
            // can definitely add it
            if(inSet(innerNodeSet_, v) ){
                // starting vertex needs to be added
                if(v==startVertex_)
                    return true;
                else{
                    // inner border needs to be added
                    if(inSet(innerBorder_,v) )
                        return true;
                    else
                        return false;
                }
            }
            else
                return true;
        }
        bool edge(const Value e) const
        {   
            const auto eInLifted = edgesInLiftedGraph_[e];
            if(currentEdgeLabels_[e]==1){
                return false;
            }
            else{
                const auto u = model_.liftedGraph().vertexOfEdge(eInLifted, 0);
                const auto v = model_.liftedGraph().vertexOfEdge(eInLifted, 1);

                // case 1 : both in inner subgraph  => false
                if(inSet(innerNodeSet_,u) && inSet(innerNodeSet_,v)){
                    return false;
                }
                else{
                    return true;
                }
            }
        }
        const LiftedMcModel & model_;
        const Set & innerNodeSet_;
        const Set & totalNodeSet_;
        const Set & innerBorder_;
        const Set & outerBorder_;
        const std::vector<size_t > & edgesInLiftedGraph_;
        const std::vector<uint8_t> & currentEdgeLabels_;
        size_t startVertex_;
    };
    struct BoundaryConditionMask2 {

        BoundaryConditionMask2(
            const LiftedMcModel & model,
            const Set &     innerNodeSet,
            const Set &     totalNodeSet,
            const Set &     innerBorder,
            const Set &     outerBorder,
            const std::vector<size_t> & edgesInLiftedGraph,
            const std::vector<uint8_t> & currentEdgeLabels
        )
        :   
        model_(model),
        innerNodeSet_(innerNodeSet),
        totalNodeSet_(totalNodeSet),
        innerBorder_(innerBorder),
        outerBorder_(outerBorder),
        edgesInLiftedGraph_(edgesInLiftedGraph),
        currentEdgeLabels_(currentEdgeLabels),
        startVertex_()
        {

        }

        typedef std::size_t Value;

        bool vertex(const Value v) const
        {   
            bool ret;
            if(inSet(totalNodeSet_, v) ){
                // starting vertex needs to be added
                if(v==startVertex_)
                    ret = true;
                else{
                    if(inSet(outerBorder_,v) ){

                        ret = true;
                    }
                    else
                        ret = false;
                }
            }
            else
                ret = true;
            //std::cout<<"mask v "<<v<<" "<<ret<<"\n";
            return ret;
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
                if(inSet(innerNodeSet_,u) && inSet(innerNodeSet_,v)){
                    return false;
                }
                else{
                    return true;
                }
            }
        }
        const LiftedMcModel & model_;
        const Set & innerNodeSet_;
        const Set & totalNodeSet_;
        const Set & innerBorder_;
        const Set & outerBorder_;
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
        auto getCenterVar = [&](){ return uintDist(generator); };


        // grow subgraph
        auto getSubgraph = [&](const size_t centerVertex){
            return std::move(verticesInGraphNeigborhood(model_.originalGraph(),
                centerVertex, settings_.subgraphRadius));
        };

        // get subgraph border w.r.t. orginal graph
        auto getSubgraphBorders = [&](const Set  & subgraphNodeSet){
            std::pair<Set, Set> ret;
            for(const auto node : subgraphNodeSet){
                for(auto aIter = originalGraph.adjacenciesFromVertexBegin(node); aIter != originalGraph.adjacenciesFromVertexEnd(node); ++aIter){
                    const auto a = *aIter;
                    const auto edge = a.edge();
                    const auto otherNode  = a.vertex();
                    if(subgraphNodeSet.find(otherNode)==subgraphNodeSet.end()){
                        ret.first.insert(node);
                        ret.second.insert(otherNode);
                    }
                }
            }
            return std::move(ret);
        };

        std::vector<uint8_t> currentEdgeLabels(liftedGraph.numberOfEdges(), 1);
        std::vector<uint64_t> currentNodeLabels(originalGraph.numberOfVertices(), 0);
        std::vector<uint64_t> nodeToSubgraphNode(model_.originalGraph().numberOfVertices());

        std::copy(labelsIn.begin(), labelsIn.end(), currentEdgeLabels.begin());

        std::vector<std::size_t> edgesInLiftedGraph(originalGraph.numberOfEdges());
        for (std::size_t i = 0; i < originalGraph.numberOfEdges(); ++i){
            auto v0 = originalGraph.vertexOfEdge(i, 0);
            auto v1 = originalGraph.vertexOfEdge(i, 1);
            edgesInLiftedGraph[i] = liftedGraph.findEdge(v0, v1).second;
        }




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




        // outer iterations
        for(auto i=0; i<settings_.numberOfIterations; ++i){


            // get node labels
            auto ccLabels = getNodeLabels(currentEdgeLabels, edgesInLiftedGraph);


            // get random subgraph center
            const auto centerNode = getCenterVar();
            const auto subgraphInnerNodeSet  = getSubgraph(centerNode);
            const auto borders = getSubgraphBorders(subgraphInnerNodeSet);
            const auto & innerBorder = borders.first;
            const auto & outerBorder = borders.second;
            auto subgraphNodeSet = subgraphInnerNodeSet;
            subgraphNodeSet.insert(outerBorder.begin(), outerBorder.end());


            if(true){//settings_.verbose >=2){
                printer("cc",[&](size_t var){
                    std::cout.width(3);
                    std::cout<<ccLabels[var]<<"  ";
                });

                printer("borders",[&](size_t var){
                    std::cout.width(3);
                    if(inSet(subgraphNodeSet,var))
                        if(inSet(innerBorder,var))
                            std::cout<<"2  ";
                        else if(inSet(outerBorder,var))
                            std::cout<<"3  ";
                        else
                            std::cout<<"1  ";
                    else
                        std::cout<<"0  ";
                });
            }   


            // build the submodel
            Graph<> subOriginalGraph(subgraphNodeSet.size());
            Graph<> subLiftedGraph(subgraphNodeSet.size());
            std::vector<float>      subEdgeCosts;
            std::vector<uint8_t>    subLabelsIn;


            std::vector<uint64_t> subGraphToGraphNode(subgraphNodeSet.size());
            auto subGraphVertex = 0;
            for(auto node : subgraphNodeSet){
                subGraphToGraphNode[subGraphVertex] = node;
                nodeToSubgraphNode[node]  = subGraphVertex;
                ++subGraphVertex;
            }
           
            //std::cout<<"    build subgraph"<<"\n";
            Set addedEdges;
            // build the subgraphs and sub edgeCosts
            for(auto node : subgraphNodeSet){
                const auto subGraphNode = nodeToSubgraphNode[node];
                // iterate over all lifted edges
                for(auto aIter = liftedGraph.adjacenciesFromVertexBegin(node); aIter != liftedGraph.adjacenciesFromVertexEnd(node); ++aIter){
                    const auto a = *aIter;
                    const auto edge = a.edge();
                    const auto otherNode  = a.vertex();

                    // other node is also in the subgraph?
                    if(subgraphNodeSet.find(otherNode) != subgraphNodeSet.end()){
                        // edge has not been added yet
                        if(addedEdges.find(edge) == addedEdges.end()){
                            const auto subGraphOtherNode = nodeToSubgraphNode[otherNode];
                            subLiftedGraph.insertEdge(subGraphNode,subGraphOtherNode);
                            subEdgeCosts.push_back(edgeCosts[edge]);
                            subLabelsIn.push_back(currentEdgeLabels[edge]);
                            // is this edge also in original graph?
                            if(originalGraph.findEdge(node,otherNode).first){
                                subOriginalGraph.insertEdge(subGraphNode,subGraphOtherNode);
                            }
                            addedEdges.insert(edge);
                        }
                    }
                }   
            }

            const auto boundaryCondEdgeStart = subLiftedGraph.numberOfEdges();


            BoundaryConditionMask2 mask(model_, subgraphInnerNodeSet,subgraphNodeSet,
                                       innerBorder, outerBorder, edgesInLiftedGraph, 
                                       currentEdgeLabels);





            

            std::map<size_t, uint8_t> lrEdges;
            const auto & border  = outerBorder;
            if(border.size() >=2){

                for(const auto startVertex : border){
                    const auto u = startVertex;
                    mask.startVertex_ = startVertex;
                    //std::cout<<"\n\n"<<startVertex<<"\n";
                    Set reachedBorderNodes;
                    // the callback
                    auto callback = [&](const size_t node, const size_t depth, 
                                        bool & proceed,bool & add ){  
                        proceed = true;
                        add = true;
                            
                        if(node != startVertex && border.find(node) != border.end()){
                            GRAPH_CHECK(subgraphNodeSet.find(node)!=subgraphNodeSet.end(),"");
                            reachedBorderNodes.insert(node);
                            add = false;
                            if(reachedBorderNodes.size() == border.size()-1)
                                proceed = false;
                        }
                    };
                    breadthFirstSearch(originalGraph, mask, startVertex, callback);

                    //std::cout<<"reachedBorderNodes" <<reachedBorderNodes.size()<<"\n";
                    for(const auto v : border){
                        if(u<v){

                            uint8_t state = inSet(reachedBorderNodes, v) ? 0 : 1;
                            if(state == 0){
                                auto cost = settings_.softInf;
                                if(state == 1){
                                    cost *= -1.0;
                                }

                                const auto subU = nodeToSubgraphNode[u];
                                const auto subV = nodeToSubgraphNode[v];
                                GRAPH_CHECK_OP(subU,<,subOriginalGraph.numberOfVertices(),"");
                                GRAPH_CHECK_OP(subV,<,subOriginalGraph.numberOfVertices(),"");

                                const auto fe = subLiftedGraph.findEdge(subU,subV);
                                const auto feo = subOriginalGraph.findEdge(subU,subV);
                                if(fe.first){
                                    lrEdges[fe.second] = state;
                                    subEdgeCosts[fe.second] = cost;
                                    subLabelsIn[fe.second] = state;
                                    if(!feo.first)
                                        subOriginalGraph.insertEdge(subU,subV);
                                }
                                else{
                                    GRAPH_CHECK(!feo.first,"");
                                    subOriginalGraph.insertEdge(subU,subV);
                                    lrEdges[subLiftedGraph.insertEdge(subU,subV)] = state;
                                    subEdgeCosts.push_back(cost);
                                    subLabelsIn.push_back(state);
                                }
                            }
                        }
                    }
                }
            }

            std::map<size_t, uint8_t> bcEdges;
            for(auto u : outerBorder){
                const auto uSub = nodeToSubgraphNode[u];
                // iterate over all lifted edges
                for(auto aIter = liftedGraph.adjacenciesFromVertexBegin(u); aIter != liftedGraph.adjacenciesFromVertexEnd(u); ++aIter){
                    const auto a = *aIter;
                    const auto edge = a.edge();
                    const auto v  = a.vertex();
                    const auto state = currentEdgeLabels[edge];
                    if(originalGraph.findEdge(u,v).first){
                        if( inSet(subgraphNodeSet, v) ){
                            const auto vSub = nodeToSubgraphNode[v];
                            const auto fsg = subLiftedGraph.findEdge(uSub, vSub);
                            GRAPH_CHECK(fsg.first,"");
                            bcEdges[fsg.second] = state;
                            auto cost = settings_.softInf;
                            if(state == 1){
                                cost *= -1.0;
                            }
                            subLabelsIn[fsg.second] = state;
                            subEdgeCosts[fsg.second] = cost;
                        }
                    }
                }
            }




            if(true){//settings_.verbose>=2){
                std::cout<<"subLiftedGraph.numberOfEdges() "<<subLiftedGraph.numberOfEdges()<<"\n";
                std::cout<<"subOriginalGraph.numberOfEdges() "<<subOriginalGraph.numberOfEdges()<<"\n";
                std::cout<<"number of must link "<<lrEdges.size()<<"\n";
                std::cout<<"number of bc edge "<<bcEdges.size()<<"\n";
            }
            //std::cout<<"constraint edges "<<subgraphLifted.numberOfEdges() - boundaryCondEdgeStart;
            std::vector<uint8_t> subLabelsOut(subLabelsIn.size());
  
            //std::cout<<"    optimize submodel"<<"\n";
            // optimize the submodel
            typedef LiftedMcModelView<Graph<>, Graph<>, std::vector<float> > SubModel;
            SubModel subModel(subOriginalGraph, subLiftedGraph, subEdgeCosts);
            solveSubProblem(subModel, subLabelsIn, subLabelsOut);
        
                
            bool violated = false;

            // violated must link ? 
            //for(auto eSub : mustLinkLiftedEdges){
            //    if(subLabelsOut[eSub]==1){
            //        violated = true;
            //        break;
            //    }
            //}

            // violated bc edges
            for(auto eSub : lrEdges){
                    if(int(subLabelsOut[eSub.first]) != int(eSub.second)){
                        violated = true;
                        break;
                    }
                }
            if(!violated){
                for(auto eSub : bcEdges){
                    if(int(subLabelsOut[eSub.first]) != int(eSub.second)){
                        violated = true;
                        break;
                    }
                }
            }

            if(!violated){
                std::cout<<"SUBGRAPH RES\n";
                auto clBuffer = currentEdgeLabels;



                for(auto subE=0; subE<boundaryCondEdgeStart; ++subE){
                    const auto subU = subLiftedGraph.vertexOfEdge(subE, 0);
                    const auto subV = subLiftedGraph.vertexOfEdge(subE, 1);
                    const auto u = subGraphToGraphNode[subU];
                    const auto v = subGraphToGraphNode[subV];

                    GRAPH_CHECK_OP(u,<,originalGraph.numberOfVertices(),"");
                    GRAPH_CHECK_OP(v,<,originalGraph.numberOfVertices(),"");
                    const auto fe = liftedGraph.findEdge(u,v);
                    GRAPH_CHECK(fe.first,"");
                    clBuffer[fe.second] = subLabelsOut[subE];
                    if(originalGraph.findEdge(u,v).first)
                        std::cout<<"u v"<<u<<" "<<v<<" = "<<int(subLabelsOut[subE])<<"\n";
                }
                auto newEnergy = getEnergy(clBuffer);
                if(newEnergy<getEnergy(currentEdgeLabels)){
                    currentEdgeLabels  =clBuffer;
                }
                if(settings_.verbose >=1)
                    std::cout<<"Energy "<<getEnergy(currentEdgeLabels)<<"\n";
            }
            else{
                //if(settings_.verbose >=1)
                //    std::cout<<"violated\n";
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
                    std::cout<<"E L "<<eSolL<<"\n";
                    std::cout<<"E 0 "<<eSol0<<"\n";
                    std::cout<<"E 1 "<<eSol1<<"\n";

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
            if (static_cast<bool>(labels[edge]) != !components.areConnected(v0, v1))
                throw std::runtime_error("the current multicut labeling is invalid.");
        }   
        return std::move(components.labels_);
    }

    const LiftedMcModel & model_;
    Settings settings_;


};





}
}
}

#endif
