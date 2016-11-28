#pragma once
#ifndef ANDRES_GRAPH_MULTICUT_ILP_HXX
#define ANDRES_GRAPH_MULTICUT_ILP_HXX

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <deque>
#include <array>
#include <algorithm>

#include "andres/partition.hxx" 
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/paths.hxx"
#include "andres/graph/components.hxx"
#include "andres/graph/shortest-paths.hxx"

namespace andres {
namespace graph {
namespace multicut {

/// Algorithm for the Minimum Cost Multicut Problem.
///
/// A variant of the algorithm proposed in
///
/// Andres B., Kroeger T., Briggman K. L., Denk W., Korogod N., Knott G.,
/// Koethe U. and Hamprecht F. A. Globally Optimal Closed-surface Segmentation
/// for Connectomics. ECCV 2012. http://dx.doi.org/10.1007/978-3-642-33712-3_56
///
template<typename ILP, typename GRAPH, typename ECA, typename ELA, typename VIS>
inline void
ilp(
    const GRAPH& graph,
    const ECA& edgeCosts,
    const ELA& inputLabels,
    ELA& outputLabels,
    VIS& visitor,
    size_t numberOfIterations = std::numeric_limits<size_t>::max()
) {
    struct SubgraphWithCut {
        SubgraphWithCut(const ILP& ilp)
            : ilp_(ilp) 
            {}
        bool vertex(const size_t v) const
            { return true; }
        bool edge(const size_t e) const
            { return ilp_.label(e) == 0; }
        const ILP& ilp_;
    };

    ComponentsBySearch<GRAPH> components;
    ILP ilp;
    std::deque<size_t> path;
    std::vector<ptrdiff_t> buffer;
    std::vector<double> variables(graph.numberOfEdges());
    std::vector<double> coefficients(graph.numberOfEdges());

    auto addCycleInequalities = [&] ()
    {
        components.build(graph, SubgraphWithCut(ilp));

        // search for violated non-chordal cycles and add corresp. inequalities
        size_t counter = 0;

        #pragma omp parallel for firstprivate(path, buffer, variables, coefficients), schedule(guided)
        for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) 
            if (ilp.label(edge) == 1)
            {
                auto v0 = graph.vertexOfEdge(edge, 0);
                auto v1 = graph.vertexOfEdge(edge, 1);

                if (components.areConnected(v0, v1))
                { 
                    // search for shortest path
                    spsp(graph, SubgraphWithCut(ilp), v0, v1, path, buffer);
                    
                    // skip chordal paths
                    if (findChord(graph, path.begin(), path.end(), true).first)
                        continue;

                    // add inequality
                    auto sz = path.size();

                    for (size_t j = 0; j < sz - 1; ++j)
                    {
                        variables[j] = static_cast<double>(graph.findEdge(path[j], path[j + 1]).second);
                        coefficients[j] = 1.0;
                    }

                    variables[sz-1] = static_cast<double>(edge);
                    coefficients[sz-1] = -1.0;

                    #pragma omp critical
                    ilp.addConstraint(variables.begin(), variables.begin() + sz, coefficients.begin(), 0, std::numeric_limits<double>::infinity());

                    #pragma omp atomic
                    ++counter;
                }
            }

        return counter;
    };

    auto repairSolution = [&] ()
    {
        for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge)
        {
            auto v0 = graph.vertexOfEdge(edge, 0);
            auto v1 = graph.vertexOfEdge(edge, 1);

            outputLabels[edge] = components.areConnected(v0, v1) ? 0 : 1;
        }

        ilp.setStart(outputLabels.begin());
    };

    ilp.initModel(graph.numberOfEdges(), edgeCosts.data());
    ilp.setStart(inputLabels.begin());

    for (size_t i = 0; numberOfIterations == 0 || i < numberOfIterations; ++i)
    {
        if (i != 0)
        {
            repairSolution();

            if (!visitor(outputLabels))
                break;
        }

        ilp.optimize();

        if (addCycleInequalities() == 0)
            break;
    }

    repairSolution();
}

template<typename ILP, typename GRAPH, typename ECA, typename ELA>
inline void
ilp(
    const GRAPH& graph,
    const ECA& edgeCosts,
    const ELA& inputLabels,
    ELA& outputLabels,
    size_t numberOfIterations = std::numeric_limits<size_t>::max()
) {
    struct Visitor {
        bool operator()(ELA const& edge_labels) const {
            return true;
        }
    } visitor;

    ilp<ILP>(graph, edgeCosts, inputLabels, outputLabels, visitor, numberOfIterations);
}

/// Algorithm for the Set Partition Problem.
///
/// The Set Partition Problem is the Minimum Cost Multicut Problem for complete
/// graphs.
///
template<typename ILP, typename GRAPH_VISITOR, typename ECA, typename ELA, typename VIS>
inline void
ilp(
    const CompleteGraph<GRAPH_VISITOR>& graph,
    const ECA& edgeCosts,
    const ELA& inputLabels,
    ELA& outputLabels,
    VIS& visitor,
    size_t numberOfIterations = std::numeric_limits<size_t>::max()
) {
    struct SubgraphWithCut {
        SubgraphWithCut(const ILP& ilp)
            : ilp_(ilp) 
            {}
        bool vertex(const size_t v) const
            { return true; }
        bool edge(const size_t e) const
            { return ilp_.label(e) == 0; }
        const ILP& ilp_;
    };

    ComponentsBySearch<CompleteGraph<GRAPH_VISITOR>> components;
    ILP ilp;
    std::array<double, 3> variables;
    std::array<double, 3> coefficients;

    auto addCycleInequalities = [&] ()
    {
        components.build(graph, SubgraphWithCut(ilp));

        size_t counter = 0;

        #pragma omp parallel for firstprivate(variables, coefficients), schedule(guided)
        for(size_t edge = 0; edge < graph.numberOfEdges(); ++edge) 
            if (ilp.label(edge) == 1)
            {
                variables[2] = edge;

                auto v0 = graph.vertexOfEdge(edge, 0);
                auto v1 = graph.vertexOfEdge(edge, 1);

                for (size_t i = 0; i < graph.numberOfVertices(); ++i)
                {
                    if (i == v0 || i == v1)
                        continue;

                    variables[0] = graph.findEdge(v0, i).second;
                    variables[1] = graph.findEdge(v1, i).second;

                    if (ilp.label(variables[0]) == 0 && ilp.label(variables[1]) == 0)
                    {
                        coefficients[0] =  1.0;
                        coefficients[1] =  1.0;
                        coefficients[2] = -1.0;

                        #pragma omp critical
                        ilp.addConstraint(variables.begin(), variables.end(), coefficients.begin(), 0, std::numeric_limits<double>::infinity());

                        #pragma omp atomic
                        ++counter;
                    }
                }
            }

        return counter;
    };

    auto repairSolution = [&] ()
    {
        for(size_t edge = 0; edge < graph.numberOfEdges(); ++edge)
        {
            auto v0 = graph.vertexOfEdge(edge, 0);
            auto v1 = graph.vertexOfEdge(edge, 1);

            outputLabels[edge] = components.areConnected(v0, v1) ? 0 : 1;
        }

        ilp.setStart(outputLabels.begin());
    };

    ilp.initModel(graph.numberOfEdges(), edgeCosts.data());
    ilp.setStart(inputLabels.begin());

    for (size_t i = 0; numberOfIterations == 0 || i < numberOfIterations; ++i)
    {
        if (i != 0)
        {
            repairSolution();

            if (!visitor(outputLabels))
                break;
        }

        ilp.optimize();

        if (addCycleInequalities() == 0)
            break;
    }

    repairSolution();
}

template<typename ILP, typename GRAPH_VISITOR, typename ECA, typename ELA>
inline void
ilp(
    const CompleteGraph<GRAPH_VISITOR>& graph,
    const ECA& edgeCosts,
    const ELA& inputLabels,
    ELA& outputLabels,
    size_t numberOfIterations = std::numeric_limits<size_t>::max()
) {
    struct Visitor
    {
        bool operator()(ELA const& edge_labels) const
        {
            return true;
        }
    } visitor;

    ilp<ILP>(graph, edgeCosts, inputLabels, outputLabels, visitor, numberOfIterations);
}

} // namespace multicut
} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_MULTICUT_ILP_HXX
