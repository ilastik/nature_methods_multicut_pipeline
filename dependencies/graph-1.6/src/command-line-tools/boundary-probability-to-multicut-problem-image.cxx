#include <stdexcept>
#include <iostream>
#include <sstream>
#include <array>
#include <fstream>
#include <tclap/CmdLine.h>

#include <andres/graph/grid-graph.hxx>
#include <andres/graph/hdf5/grid-graph.hxx>
#include <andres/functional.hxx>

#include "image.hxx"
#include "image-io.hxx"

using namespace std;
using namespace andres::graph;

typedef double value_type;
typedef size_t size_type;

struct Parameters {
    string imageFileName_;
    string HDF5FileName_;
    value_type bias_ { .5 };
};

inline void
parseCommandLine(
    int argc, 
    char** argv, 
    Parameters& parameters
) {
    try {
        TCLAP::CmdLine tclap("boundary-probability-image-to-multicut-problem", ' ', "1.0");
        TCLAP::ValueArg<string> argImageFileName("i", "input-ppm-image", "ppm image (input)", true, parameters.imageFileName_, "INPUT_PPM_IMAGE", tclap);
        TCLAP::ValueArg<string> argHDF5FileName("o", "output-hdf5-file", "hdf5 file (output)", true, parameters.HDF5FileName_, "OUTPUT_HDF5_FILE", tclap);
        TCLAP::ValueArg<value_type> argBias("b", "bias", "logistic prior probability for pixels to be cut", false, parameters.bias_, "BIAS", tclap);
        
        tclap.parse(argc, argv);

        parameters.imageFileName_ = argImageFileName.getValue();
        parameters.HDF5FileName_ = argHDF5FileName.getValue();
        parameters.bias_ = argBias.getValue();
    }
    catch(TCLAP::ArgException& e) {
        throw runtime_error(e.error());
    }
}

void
multicutProblemFromBoundaryProbabilityImageWTA(
    const Parameters& parameters
) {
    GridGraph<2> graph;
    vector<value_type> edgeCutProbabilities;

    cout << "loading image from file " << parameters.imageFileName_ << endl;
    Image inputImage;
    loadImagePPM(parameters.imageFileName_, inputImage);

    cout << "constructing graph and computing edge probabilities..." << endl;
    PixelCutProbabilityFromEdgesWTA<value_type, size_type> pixelCutProbabilityWTA(inputImage);
    constructGraphAndEdgeProbabilities(pixelCutProbabilityWTA, graph, edgeCutProbabilities, parameters.bias_);

    cout << "saving multicut problem to file: " << parameters.HDF5FileName_ << endl;

    auto file = hdf5::createFile(parameters.HDF5FileName_);
    hdf5::save(file, "graph", graph);
    hdf5::save(file, "edge-cut-probabilities", { graph.numberOfEdges() }, edgeCutProbabilities.data());
    hdf5::closeFile(file);
}
int main(int argc, char** argv) {
    try {
        Parameters parameters;
        parseCommandLine(argc, argv, parameters);

        multicutProblemFromBoundaryProbabilityImageWTA(parameters);
    }
    catch(const runtime_error& error) {
        cerr << "error: " << error.what() << endl;
        return 1;
    }

    return 0;
}

