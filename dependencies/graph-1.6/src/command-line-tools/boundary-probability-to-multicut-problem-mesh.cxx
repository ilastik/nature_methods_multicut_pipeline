#include <stdexcept>
#include <iostream>
#include <sstream>
#include <array>
#include <fstream>
#include <tclap/CmdLine.h>

#include <andres/marray.hxx>
#include <andres/graph/graph.hxx>
#include <andres/graph/hdf5/graph.hxx>
#include <andres/functional.hxx>


using namespace std;
using namespace andres::graph;

typedef double value_type;
typedef size_t size_type;
typedef vector<value_type> ValueMap;

struct Parameters {
    string binFileName_;
    string HDF5FileName_;
    value_type bias_ { .5 };
};


template<class INPUT_GRAPH>
void loadMeshProbabilities(const Parameters& parameters, INPUT_GRAPH& inputGraph, ValueMap &inputEdgeWeights) {
	value_type bias = parameters.bias_;
	value_type biasOnTheta = ::log(bias/(value_type(1)-bias));
	std::ifstream meshFile(parameters.binFileName_, std::ifstream::binary);
	int32_t j, k;
	double p;	
	if (!meshFile) std::cout << "Unable to open file "<< parameters.binFileName_<<" ";
	else
		j = 0;
	andres::graph::IdleGraphVisitor<size_type> visitor;

	int nbEdges = 0;
	int nbNodes = 0;
	do {
		meshFile.read((char *)(&j), sizeof(j));
		meshFile.read((char *)(&k), sizeof(k));
		meshFile.read((char *)(&p), sizeof(p));
		if (meshFile) {
			nbEdges++;
			nbNodes = std::max(nbNodes, j);
			nbNodes = std::max(nbNodes, k);

		} else {
			break;
		}
	} while (true);

	nbNodes++;
	inputGraph.assign(nbNodes, visitor);
	meshFile.clear();
	meshFile.seekg(0, meshFile.beg);
	do {
		j = 0;
		k = 0;
		p = 0;
		meshFile.read((char *)(&j), sizeof(j));
		meshFile.read((char *)(&k), sizeof(k));
		meshFile.read((char *)(&p), sizeof(p));
		p = std::max(p, 0.01);
		if (meshFile) {
			value_type z = ::log(std::numeric_limits<value_type>::epsilon() + p/(value_type(1) - p));
			p = value_type(1)/(value_type(1)+::exp(-z-biasOnTheta));
			const std::size_t e = inputGraph.insertEdge(static_cast<size_type>(j), static_cast<size_type>(k));		
			inputEdgeWeights.push_back(p);
		} else
			break;
	} while (true);
	meshFile.close();
	return;
}

inline void
parseCommandLine(
    int argc, 
    char** argv, 
    Parameters& parameters
)
try
{
    TCLAP::CmdLine tclap("boundary-probability-mesh-to-multicut-problem", ' ', "1.0");
    TCLAP::ValueArg<string> argBinFileName("i", "input-probabilities", ".bin object (input)", true, parameters.binFileName_, "INPUT_BIN", tclap);
    TCLAP::ValueArg<string> argHDF5FileName("o", "hdf5-file", "hdf5 file (output)", true, parameters.HDF5FileName_, "OUTPUT_HDF5_FILE", tclap);
    TCLAP::ValueArg<value_type> argBias("b", "bias", "logistic prior probability for edges to be cut", false, parameters.bias_, "BIAS", tclap);
    
    tclap.parse(argc, argv);

    parameters.binFileName_ = argBinFileName.getValue();
    parameters.HDF5FileName_ = argHDF5FileName.getValue();
    parameters.bias_ = argBias.getValue();
}
catch(TCLAP::ArgException& e) {
    throw runtime_error(e.error());
}


void multicutProblemFromBoundaryProbabilityMesh(
	const Parameters& parameters,
	ostream& stream = cerr
	)
{	
	Graph<> graph;
	ValueMap edgeCutProbabilities;

	loadMeshProbabilities(parameters, graph, edgeCutProbabilities);						

	stream << "Number of vertices: " << graph.numberOfVertices() << endl;
	stream << "Number of edges: " << graph.numberOfEdges() << endl;
	
    auto file = hdf5::createFile(parameters.HDF5FileName_);
    hdf5::save(file, "graph", graph);
    hdf5::save(file, "edge-cut-probabilities", { graph.numberOfEdges() }, edgeCutProbabilities.data());
    hdf5::closeFile(file);		
}


int main(int argc, char** argv)
try
{
    Parameters parameters;
    parseCommandLine(argc, argv, parameters);

    multicutProblemFromBoundaryProbabilityMesh(parameters);

    return 0;
}
catch(const runtime_error& error) {
    cerr << "error: " << error.what() << endl;
    return 1;
}
