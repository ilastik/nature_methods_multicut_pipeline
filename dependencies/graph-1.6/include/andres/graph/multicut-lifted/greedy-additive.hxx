#pragma once
#ifndef ANDRES_GRAPH_MULTICUT_LIFTED_GREEDY_ADDITIVE_HXX
#define ANDRES_GRAPH_MULTICUT_LIFTED_GREEDY_ADDITIVE_HXX

#include <cstddef>
#include <iterator>
#include <vector>
#include <algorithm>
#include <map>
#include <queue>

#include "andres/partition.hxx"

namespace andres {
namespace graph {
namespace multicut_lifted {

/// Graph decomposition by greedy additive edge contraction.
template<typename ORIGGRAPH, typename LIFTGRAPH, typename EVA, typename ELA>
void greedyAdditiveEdgeContraction(
    const ORIGGRAPH& original_graph,
    const LIFTGRAPH& lifted_graph,
    EVA const& edge_values,
    ELA& edge_labels,
    float nodeStopCond = -1.0
) {
    class DynamicGraph {
    public:
        DynamicGraph(size_t n)
            :   vertices_(n),
                numberOfVertices_(n)
            {}
        std::size_t numberOfVertices()const{
            return numberOfVertices_;
        }
        bool edgeExists(size_t a, size_t b) const
            {
                return !vertices_[a].empty() && vertices_[a].find(b) != vertices_[a].end();
            }
        std::map<size_t, typename EVA::value_type> const& getAdjacentVertices(size_t v) const
            {
                return vertices_[v];
            }
        typename EVA::value_type getEdgeWeight(size_t a, size_t b) const
            {
                return vertices_[a].at(b);
            }
        void removeVertex(size_t v)
            {
                for (auto& p : vertices_[v])
                    vertices_[p.first].erase(v);

                --numberOfVertices_;
                vertices_[v].clear();
            }
        void setEdgeWeight(size_t a, size_t b, typename EVA::value_type w)
            {
                vertices_[a][b] = w;
                vertices_[b][a] = w;
            }

    private:
        size_t numberOfVertices_;
        std::vector<std::map<size_t, typename EVA::value_type>> vertices_;
    };

    struct Edge {
        Edge(size_t _a, size_t _b, typename EVA::value_type _w)
            {
                if (_a > _b)
                    std::swap(_a, _b);

                a = _a;
                b = _b;

                w = _w;
            }

        size_t a;
        size_t b;
        size_t edition;
        typename EVA::value_type w;

        bool operator <(Edge const& other) const
            { return w < other.w; }
    };

    std::vector<std::map<size_t, size_t>> edge_editions(original_graph.numberOfVertices());
    DynamicGraph original_graph_cp(original_graph.numberOfVertices());
    DynamicGraph lifted_graph_cp(original_graph.numberOfVertices());
    std::priority_queue<Edge> Q;

    for (size_t i = 0; i < original_graph.numberOfEdges(); ++i) {
        auto a = original_graph.vertexOfEdge(i, 0);
        auto b = original_graph.vertexOfEdge(i, 1);
        
        original_graph_cp.setEdgeWeight(a, b, 1.);
    }

    for (size_t i = 0; i < lifted_graph.numberOfEdges(); ++i) {
        auto a = lifted_graph.vertexOfEdge(i, 0);
        auto b = lifted_graph.vertexOfEdge(i, 1);

        lifted_graph_cp.setEdgeWeight(a, b, edge_values[i]);
        
        if (original_graph_cp.edgeExists(a, b)) {
            auto e = Edge(a, b, edge_values[i]);
            e.edition = ++edge_editions[e.a][e.b];

            Q.push(e);
        }
    }

    andres::Partition<size_t> partition(original_graph.numberOfVertices());

    while (!Q.empty()) {

        if(nodeStopCond>0.0){
            int stopCond = -1;
            if(nodeStopCond<1.0)
                stopCond = double(original_graph.numberOfVertices())*nodeStopCond + 0.5;
            else
                stopCond = nodeStopCond + 0.5;
            stopCond = std::min(stopCond, int(original_graph.numberOfVertices()-1));
            stopCond = std::max(stopCond,1);
            if(original_graph_cp.numberOfVertices()<=stopCond)
                break;
        }

        auto edge = Q.top();
        Q.pop();

        if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
            continue;
        
        if (edge.w < typename EVA::value_type())
            break;

        auto stable_vertex = edge.a;
        auto merge_vertex = edge.b;

        if (lifted_graph_cp.getAdjacentVertices(stable_vertex).size() < lifted_graph_cp.getAdjacentVertices(merge_vertex).size())
            std::swap(stable_vertex, merge_vertex);

        partition.merge(stable_vertex, merge_vertex);

        for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex)) {
            if (p.first == stable_vertex)
                continue;

            original_graph_cp.setEdgeWeight(stable_vertex, p.first, 1.);
        }

        original_graph_cp.removeVertex(merge_vertex);

        for (auto& p : lifted_graph_cp.getAdjacentVertices(merge_vertex)) {
            if (p.first == stable_vertex)
                continue;

            auto t = typename EVA::value_type();

            if (lifted_graph_cp.edgeExists(stable_vertex, p.first))
                t = lifted_graph_cp.getEdgeWeight(stable_vertex, p.first);

            lifted_graph_cp.setEdgeWeight(stable_vertex, p.first, p.second + t);
            
            if (original_graph_cp.edgeExists(stable_vertex, p.first)) {
                auto e = Edge(stable_vertex, p.first, p.second + t);
                e.edition = ++edge_editions[e.a][e.b];

                Q.push(e);
            }
        }

        lifted_graph_cp.removeVertex(merge_vertex);
    }

    for (size_t i = 0; i < lifted_graph.numberOfEdges(); ++i)
        edge_labels[i] = partition.find(lifted_graph.vertexOfEdge(i, 0)) == partition.find(lifted_graph.vertexOfEdge(i, 1)) ? 0 : 1;
}

template<class LIFTED_MC_MODEL>
class GreedyAdditiveEdgeContraction{
public:
    typedef LIFTED_MC_MODEL LiftedMcModel;
    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef GreedyAdditiveEdgeContraction<OTHER_LIFTED_MC_MODEL> type;
    };

    struct Settings{
        Settings()
        :   nodeStopCond(-1.0)
        {

        }
        template<class OTHER>
        Settings(const OTHER & other){
            this->nodeStopCond = other.nodeStopCond;
        }
        float nodeStopCond;
    };

    GreedyAdditiveEdgeContraction(const LiftedMcModel & model, const Settings & settings = Settings())
    :   model_(model),
        settings_(settings){
    }

    template<class LABELS_IN, class LABELS_OUT>
    void run(const LABELS_IN & labelsIn, LABELS_OUT & labelsOut){
        greedyAdditiveEdgeContraction(model_.originalGraph(), model_.liftedGraph(), 
                     model_.edgeCosts(),labelsOut,settings_.nodeStopCond);
    }

    const LiftedMcModel & getModel()const{
        return model_;
    }
private:
    const LiftedMcModel model_;
    Settings settings_;
};





} // namespace multicut_lifted 
} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_MULTICUT_LIFTED_GREEDY_ADDITIVE_HXX
