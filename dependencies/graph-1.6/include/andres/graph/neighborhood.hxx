#pragma once
#ifndef ANDRES_GRAPH_NEIGHBORHOOD_HXX
#define ANDRES_GRAPH_NEIGHBORHOOD_HXX

#include <set>
#include "andres/graph/bfs.hxx"


namespace andres {
namespace graph {


    template<class GRAPH>
    std::set<size_t> verticesInGraphNeigborhood(
        const GRAPH & g,
        const std::size_t centerVertex,
        const std::size_t graphRadius
    ){
        std::set<size_t> nodesInSubgraph;
        auto callback = [&](
            const size_t node, 
            const size_t depth, 
            bool & proceed,
            bool & add 
        ){  
            proceed = true;
            add = true;
            if(depth >= graphRadius){
                add = false;
            }
            nodesInSubgraph.insert(node);
        };

        breadthFirstSearch(g, centerVertex, callback);
        return std::move(nodesInSubgraph);
    }


} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_NEIGHBORHOOD_HXX
