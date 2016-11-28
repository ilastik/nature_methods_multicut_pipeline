#include <stdexcept>
#include <iostream>
#include <sstream>

#include <tclap/CmdLine.h>

#include <andres/functional.hxx>

#include <andres/graph/grid-graph.hxx>
#include <andres/graph/graph.hxx>
#include <andres/graph/hdf5/grid-graph.hxx>
#include <andres/graph/hdf5/graph.hxx>
#include <andres/graph/multicut-lifted/greedy-additive.hxx>
#include <andres/graph/multicut-lifted/kernighan-lin.hxx>

#include "Timer.hpp"
#include "functional.hxx"
#include "utils.hxx"



using namespace std;
using namespace andres::graph;


enum class Method {
    GAEC,
    Kernighan_Lin,
};

enum class Initialization {
    Zeros,
    Ones,
    Tiles,
    Input_Labeling,
    GAEC
};

struct Parameters {
    string inputHDF5FileName_;
    string outputHDF5FileName_;
    string labelingHDF5FileName_;
    Method optimizationMethod_ { Method::Kernighan_Lin };
    Initialization initialization_ { Initialization::Zeros };
};

inline void
parseCommandLine(
    int argc,
    char** argv,
    Parameters& parameters
)
{
    try
    {
        TCLAP::CmdLine tclap("solve-lifted-multicut-problem-grid-graph", ' ', "1.0");
        TCLAP::ValueArg<string> argInputHDF5FileName("i", "input-hdf5-file", "File to load multicut problem from", true, parameters.inputHDF5FileName_, "INPUT_HDF5_FILE", tclap);
        TCLAP::ValueArg<string> argOutputHDF5FileName("o", "output-hdf-file", "hdf file (output)", true, parameters.outputHDF5FileName_, "OUTPUT_HDF5_FILE", tclap);
        TCLAP::ValueArg<string> argLabelingHDF5FileName("l", "labeling-hdf-file", "hdf file specifying initial node labelings (input)", false, parameters.labelingHDF5FileName_, "LABELING_HDF5_FILE", tclap);
        TCLAP::ValueArg<string> argOptimizationMethod("m", "optimization-method", "optimization method to use (GAEC=greedy additive edge contraction, KL=Kernighan-Lin)", false, "KL", "OPTIMIZATION_METHOD", tclap);
        TCLAP::ValueArg<string> argInitializationMethod("I", "initialization-method", "initialization method to use (zeros, ones, tiles, GAEC)", false, "zeros", "INITIALIZATION_METHOD", tclap);

        tclap.parse(argc, argv);

        parameters.inputHDF5FileName_ = argInputHDF5FileName.getValue();
        parameters.outputHDF5FileName_ = argOutputHDF5FileName.getValue();

        if (!argOptimizationMethod.isSet())
            throw runtime_error("No optimization method specified");
        
        if (argOptimizationMethod.getValue() == "GAEC")
            parameters.optimizationMethod_ = Method::GAEC;
        else if(argOptimizationMethod.getValue() == "KL")
        {
            parameters.optimizationMethod_ = Method::Kernighan_Lin;

            if (!argInitializationMethod.isSet() && !argLabelingHDF5FileName.isSet())
                throw runtime_error("Either initialization method (zeros, ones) or initial labeling must be specified for Kernighan-Lin.");

            if(argLabelingHDF5FileName.isSet())
            {
                if(argInitializationMethod.isSet())
                    throw runtime_error("Either initialization method or initial labeling must be specified.");
                
                parameters.labelingHDF5FileName_ = argLabelingHDF5FileName.getValue();
                parameters.initialization_ = Initialization::Input_Labeling;
            }
            else if (argInitializationMethod.isSet())
            {
                if(argInitializationMethod.getValue() == "ones")
                    parameters.initialization_ = Initialization::Ones;
                else if(argInitializationMethod.getValue() == "zeros")
                    parameters.initialization_ = Initialization::Zeros;
                else if(argInitializationMethod.getValue() == "tiles")
                    parameters.initialization_ = Initialization::Tiles;
                else if(argInitializationMethod.getValue() == "GAEC")
                    parameters.initialization_ = Initialization::GAEC;
                else
                    throw runtime_error("Invalid initialization method specified");
            }
        }
        else
            throw runtime_error("Invalid optimization method specified");
    }
    catch(TCLAP::ArgException& e)
    {
        throw runtime_error(e.error());
    }
}


void solveLiftedMulticutProblem(
    const Parameters& parameters,
    ostream& stream = cerr
)
{
    GridGraph<2> original_graph;
    Graph<> lifted_graph;
    vector<double> edge_values;

    // Load Lifted Multicut Problem
    {
        auto fileHandle = hdf5::openFile(parameters.inputHDF5FileName_);

        hdf5::load(fileHandle, "graph", original_graph);        
        hdf5::load(fileHandle, "graph-lifted", lifted_graph);

        vector<size_t> shape;
        hdf5::load(fileHandle, "edge-cut-probabilities", shape, edge_values);
        hdf5::closeFile(fileHandle);
    }

    transform(
        edge_values.begin(),
        edge_values.end(),
        edge_values.begin(),
        andres::NegativeLogProbabilityRatio<double,double>()
        );

    // Solve Lifted Multicut problem
    vector<char> edge_labels(lifted_graph.numberOfEdges());
    
    Timer t;
    t.start();

    if (parameters.initialization_ == Initialization::Zeros)
        fill(edge_labels.begin(), edge_labels.end(), 0);
    else if (parameters.initialization_ == Initialization::Ones)
        fill(edge_labels.begin(), edge_labels.end(), 1);
    else if (parameters.initialization_ == Initialization::Input_Labeling)
    {
        auto fileHandle = hdf5::openFile(parameters.labelingHDF5FileName_);

        vector<size_t> shape;
        vector<size_t> vertex_labels;
        hdf5::load(fileHandle, "labels", shape, vertex_labels);

        hdf5::closeFile(fileHandle);

        assert(shape.size() == 1);
        assert(shape[0] == lifted_graph.numberOfVertices());

        edge_labels.resize(lifted_graph.numberOfEdges());

        vertexToEdgeLabels(original_graph, lifted_graph, vertex_labels, edge_labels);
    }
    else if (parameters.initialization_ == Initialization::Tiles)
    {
        vector<size_t> vertex_labels(lifted_graph.numberOfVertices(), lifted_graph.numberOfVertices());
        
        size_t mdistance = 20;
        size_t current_label = 0;

        GridGraph<2>::VertexCoordinate cv;
        for(size_t v = 0; v < original_graph.numberOfVertices(); ++v)
        {
            if(vertex_labels[v] < lifted_graph.numberOfVertices())
                continue;

            original_graph.vertex(v, cv);
            vertex_labels[v] = current_label;

            // Add middle horizontal line
            {
                auto y = cv[1];
                auto colN = min(cv[0] + mdistance, original_graph.shape(0) - 1);

                for(auto x = cv[0]; x <= colN; ++x)
                    vertex_labels[original_graph.vertex({{x, y}})] = current_label;
            }

            // fill below window
            if (cv[1] < original_graph.shape(1) - 1)
            {
                auto rowN = min(cv[1] + mdistance, original_graph.shape(1) - 1);

                for (auto y = cv[1] + 1; y <= rowN; ++y)
                {
                    auto colN = min(cv[0] + mdistance, original_graph.shape(0) - 1);

                    for(auto x = cv[0]; x <= colN; ++x)
                        vertex_labels[original_graph.vertex({{x, y}})] = current_label;
                }
            }
      
            ++current_label;
        }
    
        edge_labels.resize(lifted_graph.numberOfEdges());

        vertexToEdgeLabels(original_graph, lifted_graph, vertex_labels, edge_labels);
    }
    else if (parameters.initialization_ == Initialization::GAEC)
        multicut_lifted::greedyAdditiveEdgeContraction(original_graph, lifted_graph, edge_values, edge_labels);

    if (parameters.optimizationMethod_ == Method::GAEC)
        multicut_lifted::greedyAdditiveEdgeContraction(original_graph, lifted_graph, edge_values, edge_labels);
    else if (parameters.optimizationMethod_ == Method::Kernighan_Lin)
        multicut_lifted::kernighanLin(original_graph, lifted_graph, edge_values, edge_labels, edge_labels);
    else
        throw runtime_error("Unsupported algorithm");
    
    t.stop();
    
    stream << "saving decomposition into file: " << parameters.outputHDF5FileName_ << endl;
    {
        auto file = hdf5::createFile(parameters.outputHDF5FileName_);
        
        hdf5::save(file, "graph", original_graph);

        vector<size_t> vertex_labels(lifted_graph.numberOfVertices());
        edgeToVertexLabels(lifted_graph, edge_labels, vertex_labels);

        hdf5::save(file, "labels", { vertex_labels.size() }, vertex_labels.data());

        auto energy_value = inner_product(edge_values.begin(), edge_values.end(), edge_labels.begin(), .0);

        hdf5::save(file, "energy-value", energy_value);
        hdf5::save(file, "running-time", t.get_elapsed_seconds());

        vector<char> true_edge_labels(lifted_graph.numberOfEdges());
        vertexToEdgeLabels(original_graph, lifted_graph, vertex_labels, true_edge_labels);

        auto true_energy_value = inner_product(edge_values.begin(), edge_values.end(), true_edge_labels.begin(), .0);        

        hdf5::save(file, "true-energy-value", true_energy_value);

        hdf5::closeFile(file);

        cout << "Number of clusters: " << *max_element(vertex_labels.begin(), vertex_labels.end()) + 1 << endl;
        cout << "Energy value: " << energy_value << endl;
        cout << "Running time: " << t.to_string() << endl;
        cout << "True energy value: " << true_energy_value << endl;
    }
}

int main(int argc, char** argv) {
    try {
        Parameters parameters;
        parseCommandLine(argc, argv, parameters);
        solveLiftedMulticutProblem(parameters);
    }
    catch(const runtime_error& error) {
        cerr << "error creating multicut problem: " << error.what() << endl;
        return 1;
    }

    return 0;
}
