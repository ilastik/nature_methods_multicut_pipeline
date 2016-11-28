//#pragma once
#ifndef ANDRES_GRAPH_MULTICUT_LIFTED_MC_MODEL_HXX
#define ANDRES_GRAPH_MULTICUT_LIFTED_MC_MODEL_HXX

#include <vector>
#include <random>
#include <iostream> 

#include "andres/runtime-check.hxx"
#include "andres/graph/components.hxx"


namespace andres {
namespace graph {
namespace multicut_lifted {

    /// lifted multicut algorithms should work on such an LiftedMcModel
    template<class MODEL, class ORIGINAL_GRAPH, class LIFTED_GRAPH, class EDGE_COSTS, class VALUE_TYPE>
    class LiftedMcModelCRTBase{
    public:
        typedef ORIGINAL_GRAPH OriginalGraph;
        typedef LIFTED_GRAPH LiftedGraph;
        typedef EDGE_COSTS EdgeCosts;
        typedef VALUE_TYPE value_type;
        
    };


    template<class ORIGINAL_GRAPH, class LIFTED_GRAPH, class EDGE_COSTS>
    class LiftedMcModelView : public LiftedMcModelCRTBase<
        LiftedMcModelView<ORIGINAL_GRAPH, LIFTED_GRAPH, EDGE_COSTS>,
        ORIGINAL_GRAPH,
        LIFTED_GRAPH,
        EDGE_COSTS,
        typename EDGE_COSTS::value_type
    >{
        public:
            typedef ORIGINAL_GRAPH OriginalGraph;
            typedef LIFTED_GRAPH LiftedGraph;
            typedef typename EDGE_COSTS::value_type ValueType;
            typedef EDGE_COSTS EdgeCosts;
            typedef LiftedMcModelView<OriginalGraph, LiftedGraph, EdgeCosts> Self;
            typedef LiftedMcModelCRTBase<Self, OriginalGraph, LiftedGraph, EdgeCosts, ValueType> Base;

            LiftedMcModelView(const OriginalGraph & originalGraph, const LiftedGraph & liftedGraph,
                              const EdgeCosts & edgeCosts)
            :   originalGraph_(originalGraph),
                liftedGraph_(liftedGraph),
                edgeCosts_(edgeCosts){
            }

            const OriginalGraph & originalGraph()const{
                return originalGraph_;
            }
            const LiftedGraph & liftedGraph()const{
                return liftedGraph_;
            }
            const EdgeCosts & edgeCosts()const{
                return edgeCosts_;
            }

        private:
            const OriginalGraph & originalGraph_;
            const LiftedGraph   & liftedGraph_;
            const EdgeCosts     & edgeCosts_;
    };


    template<class ORIGINAL_GRAPH, class VALUE_TYPE>
    class LiftedMcModel : public LiftedMcModelCRTBase<
        LiftedMcModel<ORIGINAL_GRAPH, VALUE_TYPE>,
        ORIGINAL_GRAPH,
        Graph<>,
        std::vector<VALUE_TYPE>,
        VALUE_TYPE
    > {
    public:
        typedef ORIGINAL_GRAPH OriginalGraph;
        typedef Graph<> LiftedGraph;
        typedef VALUE_TYPE ValueType;
        typedef std::vector<ValueType> EdgeCosts;
        typedef LiftedMcModel<OriginalGraph, ValueType> Self;
        typedef LiftedMcModelCRTBase<Self, OriginalGraph, LiftedGraph, EdgeCosts, ValueType> Base;

        LiftedMcModel(const ORIGINAL_GRAPH & originalGraph)
        :   Base(),
            originalGraph_(originalGraph),
            liftedGraph_(originalGraph.numberOfVertices()),
            edgeCosts_(originalGraph.numberOfEdges(),0){

            // insert orginal edges into lifted graph
            for(std::size_t e=0; e<originalGraph_.numberOfEdges(); ++e){
                auto v0 = originalGraph_.vertexOfEdge(e, 0);
                auto v1 = originalGraph_.vertexOfEdge(e, 1);
                liftedGraph_.insertEdge(v0, v1);
            }
        }
        OriginalGraph & _originalGraph(){
            return const_cast<ORIGINAL_GRAPH &>(originalGraph_);
        }
        const OriginalGraph & originalGraph()const{
            return originalGraph_;
        }
        LiftedGraph & _liftedGraph(){
            return const_cast<LiftedGraph &>(liftedGraph_);
            //return liftedGraph_;
        }
        const LiftedGraph & liftedGraph()const{
            return liftedGraph_;
        }
        EdgeCosts & _edgeCosts(){
            return const_cast<EdgeCosts &>(edgeCosts_);
            //return edgeCosts_;
        }
        const EdgeCosts & edgeCosts()const{
            return edgeCosts_;
        }

        std::size_t setCost(const size_t u, const size_t v , const ValueType c, const bool overwrite = true){
            const auto fe = liftedGraph_.findEdge(u,v);
            if(fe.first){
                //std::cout<<"a1\n";
                const auto e = fe.second;
                edgeCosts_[e] = overwrite ? c : edgeCosts_[e] + c;
                return fe.second;
            }
            else{
                //std::cout<<"a2\n";
                auto e = liftedGraph_.insertEdge(u,v);
                //std::cout<<"liftedGraphEdgeNum"<<liftedGraph_.numberOfEdges()<<"\n";
                edgeCosts_.push_back(c);
                return e;
            }
        }

        template<class U_ITER,class V_ITER, class COST_ITER>
        void setCosts(
            U_ITER uIter, 
            U_ITER uEnd, 
            V_ITER vIter, 
            COST_ITER costIter,
            const bool overwrite = true
        ){
            while(uIter!=uEnd){
                const auto u = *uIter;
                const auto v = *vIter;
                const auto c = *costIter;
                this->setCost(u, v, c, overwrite);
                ++uIter;
                ++vIter;
                ++costIter;
            }
        }

        // lifted edge to node labels
        template<class EDGE_LABELS, class NODE_LABELS>
        void getNodeLabels(const EDGE_LABELS & edgeLabels, NODE_LABELS & nodeLabels)const{

            struct SubgraphWithCut { // a subgraph with cut mask
                SubgraphWithCut(const EDGE_LABELS & labels, 
                                std::vector<std::size_t> const& edge_in_lifted_graph)
                    : labels_(labels), edge_in_lifted_graph_(edge_in_lifted_graph)
                    {}
                bool vertex(const std::size_t v) const
                    { return true; }
                bool edge(const std::size_t e) const
                    { return labels_[edge_in_lifted_graph_[e]] == 0; }

                std::vector<std::size_t> const& edge_in_lifted_graph_;
                const EDGE_LABELS & labels_;
            };
            std::vector<size_t> edgeInLiftedGraph(originalGraph_.numberOfEdges());
            for (std::size_t i = 0; i < originalGraph_.numberOfEdges(); ++i){
                auto v0 = originalGraph_.vertexOfEdge(i, 0);
                auto v1 = originalGraph_.vertexOfEdge(i, 1);
                edgeInLiftedGraph[i] = liftedGraph_.findEdge(v0, v1).second;
            } 
            ComponentsBySearch<OriginalGraph > components;
            components.build(originalGraph_, SubgraphWithCut(edgeLabels, edgeInLiftedGraph));
            for(std::size_t n=0; n<originalGraph_.numberOfVertices(); ++n){
                nodeLabels[n] = components.labels_[n];
            }
        }


        template<class NODE_LABELS, class EDGE_LABELS>
        void getEdgeLabels(const  NODE_LABELS & nodeLabels, EDGE_LABELS & edgeLabels)const{
            for(auto e=0 ;e<liftedGraph_.numberOfEdges(); ++e){
                auto v0 = liftedGraph_.vertexOfEdge(e, 0);
                auto v1 = liftedGraph_.vertexOfEdge(e, 1);
                edgeLabels[e] = (nodeLabels[v0] != nodeLabels[v1]) ? 1 : 0;
            }
        }
        
    private:
        const OriginalGraph & originalGraph_;
        LiftedGraph liftedGraph_;
        EdgeCosts  edgeCosts_;
    };








}
}
}

#endif
