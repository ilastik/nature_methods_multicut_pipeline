#include <stdexcept>
#include <iostream>
#include <fstream>
#include <numeric>

#include <tclap/CmdLine.h>

#include <andres/graph/hdf5/hdf5.hxx>


using namespace std;
using namespace andres::graph;



struct Parameters {    
    string inputHDF5FileName_;
    string inputMeshFileName_;
    string outputWRLFileName_;
};

template<typename T = double, typename S = size_t>
class Mesh {
public:
    Mesh() {};

    void saveAsVRML(const char* filename, vector<S> &faceID, bool colorPerFace = false) {

        FILE* f = fopen(filename, "w+");
        fprintf(f, "#VRML V2.0 utf8\n");
        fprintf(f, "Transform{\n");
        fprintf(f, "    scale 1 1 1\n");
        fprintf(f, "    translation 0 0 0\n");
        fprintf(f, "    children\n");
        fprintf(f, "    [\n");
        fprintf(f, "        Shape\n");
        fprintf(f, "        {\n");
        if (colorPerFace) {
            fprintf(f, "        appearance Appearance{\n");
            fprintf(f, "            material Material{}\n");
            fprintf(f, "        }\n");
        }
        fprintf(f, "    geometry IndexedFaceSet\n");
        fprintf(f, "    {\n");
        fprintf(f, "            creaseAngle .5\n");
        fprintf(f, "            solid TRUE\n");
        fprintf(f, "            coord Coordinate\n");
        fprintf(f, "            {\n");
        fprintf(f, "        point\n");
        fprintf(f, "        [\n");

        for (int i=0; i<vertices.size()/3; i++) {

            for (int j=0; j<2; j++) {
                fprintf(f, "%f ", vertices[i*3+j]);
            }
            if (i!=vertices.size()/3-1)
                fprintf(f, "%f, ", vertices[i*3+2]);
            else
                fprintf(f, "%f\n ]\n }\n", vertices[i*3+2]);
        }

        if (colorPerFace) {
            fprintf(f, "colorPerVertex FALSE\n");
        } else {
            fprintf(f, "colorPerVertex TRUE\n");
        }
        fprintf(f, "color Color\n");
        fprintf(f, "{\n");
        fprintf(f, "    color\n");
        fprintf(f, "    [\n");

        for (int i=0; i<faceID.size(); i++) {
            long int id = faceID[i];
            if (i!=faceID.size()-1)
                fprintf(f, "%f %f %f, \n", ((id*id*1012+4)%251)/251., ((id*id*3018+91)%259)/259., ((id*id*762+75)%181)/181.);
            else
                fprintf(f, "%f %f %f \n]\n }\n", ((id*id*1012+4)%251)/251., ((id*id*3018+91)%259)/259., ((id*id*762+75)%181)/181.);

        }

        fprintf(f, "coordIndex\n");
        fprintf(f, "    [\n");
        for (int i=0; i<faces.size()/3; i++) {
            if (i!=faces.size()/3-1)
                fprintf(f, "%u,%u,%u,-1,", faces[i*3+0], faces[i*3+1], faces[i*3+2]);
            else
                fprintf(f, "%u,%u,%u,-1\n] \n } \n", faces[i*3+0], faces[i*3+1], faces[i*3+2]);

        }

        fprintf(f, "appearance Appearance\n");
        fprintf(f, "{\n");
        fprintf(f, "    material Material\n");
        fprintf(f, "    {\n");
        fprintf(f, "    ambientIntensity 0.2\n");
        fprintf(f, "    diffuseColor 0.9 0.9 0.9\n");
        fprintf(f, "    specularColor .1 .1 .1\n");
        fprintf(f, "    shininess .5\n");
        fprintf(f, "}\n");
        fprintf(f, "}\n");
        fprintf(f, "}\n");
        fprintf(f, "]\n");
        fprintf(f, "}\n");
        fclose(f);

    }

    void readFromOFF(const char* filename) {
        FILE* f = fopen(filename, "r+");

        const char* s[4];
        fscanf(f, "%s\n", s); // OFF
        int nbvtx, nbfaces, nbxx;
        fscanf(f, "%u %u %u\n", &nbvtx, &nbfaces, &nbxx);

        vertices.resize(nbvtx*3);
        faces.resize(nbfaces*3);
        for (int i=0; i<nbvtx; i++) {
            float x, y, z;
            fscanf(f, "%f %f %f\n", &x, &y, &z);
            vertices[i*3] = x;
            vertices[i*3+1] = y;
            vertices[i*3+2] = z;
        }
        for (int i=0; i<nbfaces; i++) {
            int a, b, c;
            int nbv;
            fscanf(f, "%u %u %u %u\n", &nbv, &a, &b, &c);
            assert(nbv==3);
            faces[i*3] = a;
            faces[i*3+1] = b;
            faces[i*3+2] = c;
        }

        fclose(f);
    }


    vector<T> vertices;
    vector<S> faces;
};

inline void
parseCommandLine(
    int argc,
    char** argv,
    Parameters& parameters
) {
    try {
        TCLAP::CmdLine tclap("plot-solution-mesh", ' ', "1.0");
        TCLAP::ValueArg<string> argInputGraphHDF5FileName("i", "input-graph", "File to load graph and vertex labeling from", true, parameters.inputHDF5FileName_, "INPUT_HDF5_FILE", tclap);
        TCLAP::ValueArg<string> argInputMeshFileName("n", "input-mesh", "File to load mesh from", true, parameters.inputMeshFileName_, "INPUT_MESH_FILE", tclap);
        TCLAP::ValueArg<string> argOutputWRLFileName("o", "output", "wrl file (output)", true, parameters.outputWRLFileName_, "OUTPUT_WRL_FILE", tclap);

        tclap.parse(argc, argv);

        parameters.inputHDF5FileName_ = argInputGraphHDF5FileName.getValue();
        parameters.inputMeshFileName_ = argInputMeshFileName.getValue();
        parameters.outputWRLFileName_ = argOutputWRLFileName.getValue();
    }
    catch(TCLAP::ArgException& e) {
        throw runtime_error(e.error());
    }
}

void plotNodeLabelingGridGraph(
    const Parameters& parameters,
    ostream& stream = cerr
) {
    vector<size_t> nodeLabeling;
    Mesh<double> offMesh;

    stream << "loading labeling from HDF5 file: "<< parameters.inputHDF5FileName_ << endl;
    {
        hid_t fileHandle = hdf5::openFile(parameters.inputHDF5FileName_);

        vector<size_t> shape;
        hdf5::load(fileHandle, "labels", shape, nodeLabeling);
        
        hdf5::closeFile(fileHandle);

        assert(shape.size()==1);
        assert(shape[0] == graph.numberOfVertices());

        offMesh.readFromOFF(parameters.inputMeshFileName_.c_str());
    }
    
    stream << "saving decomposition to WRL file: " << parameters.outputWRLFileName_ << endl;

    offMesh.saveAsVRML(parameters.outputWRLFileName_.c_str(), nodeLabeling, true);
}

int main(int argc, char** argv)
try
{
    Parameters parameters;
    parseCommandLine(argc, argv, parameters);
    plotNodeLabelingGridGraph(parameters);

    return 0;
}
catch(const runtime_error& error) {
    cerr << "error plotting solution: " << error.what() << endl;
    return 1;
}
