#include <stdexcept>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>

#include <tclap/CmdLine.h>

#include <andres/graph/grid-graph.hxx>
#include <andres/graph/hdf5/grid-graph.hxx>

#include "functional.hxx"
#include "svg.hxx"
#include "image-io.hxx"


using namespace std;
using namespace andres::graph;


struct Parameters {
    string inputHDF5FileName;
    string outputSVGFileName;
    string outputPPMFileName;
    string imageFileName;
};

inline void
parseCommandLine(
    int argc,
    char** argv,
    Parameters& parameters
) 
try
{
    TCLAP::CmdLine tclap("plot-solution-image", ' ', "1.0");
    TCLAP::ValueArg<string> argInputGraphHDF5FileName("i", "input-file", "File to load graph and vertex labeling from", true, parameters.inputHDF5FileName, "INPUT_HDF5_FILE", tclap);
    TCLAP::ValueArg<string> argOutputSVGFileName("s", "svg-output", "svg with edges", false, parameters.outputSVGFileName, "OUTPUT_SVG_FILE", tclap);
    TCLAP::ValueArg<string> argOutputPPMFileName("p", "ppm-output", "svg with node colors", false, parameters.outputPPMFileName, "OUTPUT_PPM_FILE", tclap);
    TCLAP::ValueArg<string> argOriginalFileName("l", "image", "original image to draw boundaries over", false, parameters.imageFileName, "ORIGINAL_IMAGE_FILE", tclap);

    tclap.parse(argc, argv);

    parameters.inputHDF5FileName = argInputGraphHDF5FileName.getValue();
    parameters.outputSVGFileName = argOutputSVGFileName.getValue();
    parameters.outputPPMFileName = argOutputPPMFileName.getValue();
    parameters.imageFileName = argOriginalFileName.getValue();

    if (parameters.outputSVGFileName.empty() && parameters.outputPPMFileName.empty())
        throw runtime_error("At least .svg or .ppm file names must be specified");
}
catch(TCLAP::ArgException& e) {
    throw runtime_error(e.error());
}

class DecompositionFromNodeLabelingGetter{
public:
    DecompositionFromNodeLabelingGetter(
        GridGraph<2>& graph,
        vector<size_t>& nodeLabeling
    )   : graph_(graph), nodeLabeling_(nodeLabeling) {}
    int getEdgeCutProbability(const GridGraph<2>::EdgeCoordinate& eCoord) const {
        auto e = graph_.edge(eCoord);
        return nodeLabeling_[graph_.vertexOfEdge(e, 0)] != nodeLabeling_[graph_.vertexOfEdge(e, 1)] ? 1 : 0;
    }
    int up(size_t x, size_t y) const {
        return getEdgeCutProbability(GridGraph<2>::EdgeCoordinate({{x,y}}, 1));
    }
    int right(size_t x, size_t y) const {
        return getEdgeCutProbability(GridGraph<2>::EdgeCoordinate({{x,y}}, 0));
    }

private:
    const vector<size_t>& nodeLabeling_;
    const GridGraph<2>& graph_;
};

void plotNodeLabelingGridGraph(
    const Parameters& parameters,
    ostream& stream = cerr
) {
    GridGraph<2> graph;
    vector<size_t> vertex_labels;

    stream << "loading Grid Graph from HDF5 file: "<< parameters.inputHDF5FileName << endl;
    hid_t fileHandle = hdf5::openFile(parameters.inputHDF5FileName);

    hdf5::load(fileHandle, "graph", graph);

    vector<size_t> shape;
    hdf5::load(fileHandle, "labels", shape, vertex_labels);
    
    hdf5::closeFile(fileHandle);

    assert(shape.size()==1);
    assert(shape[0] == graph.numberOfVertices());

    auto W = graph.shape(0);
    auto H = graph.shape(1);
    
    if (!parameters.outputSVGFileName.empty())
    {
        typedef SavingSVG<int> SavingSVG;
        SavingSVG savingSVG(1, 1);

        DecompositionFromNodeLabelingGetter decompositionFromNodeLabelingGetter (graph, vertex_labels);
        
        stream << "saving decomposition to SVG file: " << parameters.outputSVGFileName << endl;
        
        ofstream fstream(parameters.outputSVGFileName);
        
        struct red
        {
            void operator()(std::array<unsigned char, 3>& RGB, const int prob) const
            {
                RGB[0] = 255;
                RGB[1] = RGB[2] = 0;
            }
        };

        savingSVG(decompositionFromNodeLabelingGetter, W, H, fstream, SavingSVG::ContourPathDrawer(), red(), parameters.imageFileName);
        
        fstream.close();
    }

    if (!parameters.outputPPMFileName.empty())
    {        
        auto N = *max_element(vertex_labels.begin(), vertex_labels.end()) + 1;

        map<size_t, size_t> colors;
        for (size_t i = 0; i < N; ++i)
            colors[i] = (i + 1) * (1 << 24) / (N + 1);

        vector<unsigned char> image(W*H*3);

        for (int i = 0; i < W*H; ++i)
            convertColor(colors[vertex_labels[i]], &image[i*3]);

        writePPM(parameters.outputPPMFileName.c_str(), W, H, image.data());
    }
}

int main(int argc, char** argv) {
    try {
        Parameters parameters;
        parseCommandLine(argc, argv, parameters);
        plotNodeLabelingGridGraph(parameters);
    }
    catch(const runtime_error& error) {
        cerr << "error plotting solution: " << error.what() << endl;
        return 1;
    }

    return 0;
}
