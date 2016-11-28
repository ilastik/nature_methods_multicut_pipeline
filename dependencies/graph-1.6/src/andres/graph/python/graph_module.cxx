// define PY_ARRAY_UNIQUE_SYMBOL (required by the numpy C-API)
#define PY_ARRAY_UNIQUE_SYMBOL andres_graph_PyArray_API

// include the vigranumpy C++ API
#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>




void exportGridGraph2d();
void exportGridGraph3d();
void exportGraph();
void exportLiftedMc();
void exportLiftedMcModel();
void exportedParallelLiftedMc();
void exportLiftedGa();
void exportLiftedKL();
void exportLearnLifted();

// the argument of the init macro must be the module name
BOOST_PYTHON_MODULE_INIT(_graph)
{
    // initialize numpy and vigranumpy
    vigra::import_vigranumpy();


    exportGridGraph2d();
    exportGridGraph3d();
    exportGraph();
    exportLiftedMc();
    exportLiftedMcModel();
    exportedParallelLiftedMc();
    exportLiftedGa();
    exportLiftedKL();
    exportLearnLifted();
}
