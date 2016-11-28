cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
project(graph)
include_directories(include)
file(GLOB headers
    include/andres/*.hxx
    include/andres/ilp/*.hxx
    include/andres/graph/*.hxx
    include/andres/graph/multicut/*.hxx
    include/andres/graph/multicut-lifted/*.hxx
    include/andres/graph/twocut-lifted/*.hxx
    include/andres/graph/hdf5/*.hxx
    include/andres/graph/detail/*.hxx
)
enable_testing()

##############################################################################
# Doxygen
##############################################################################
find_package(Doxygen)

##############################################################################
# OpenMP
##############################################################################
#find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

##############################################################################
# HDF5
##############################################################################
if(WIN32)
 # FindHDF5 is broken on Windows
 # it wrongly discovers the dlls instead of the lib files
 # see: www.cmake.org/Bug/bug_relationship_graph.php?bug_id=14111
 # therefore we set the variable manually
 find_library( HDF5_IMPORT_LIB NAMES hdf5 hdf5dll )
 find_library( HDF5_HL_IMPORT_LIB NAMES hdf5_hl hdf5_hldll )
 find_path(HDF5_INCLUDE_DIR hdf5.h)
 set( HDF5_LIBRARIES ${HDF5_IMPORT_LIB} ${HDF5_HL_IMPORT_LIB} )
 message(STATUS "HDF5: autodiscovery is broken on WIN32; using global search paths")
 message(STATUS "HDF5: found ${HDF5_LIBRARIES}")
else()
 find_package(HDF5 1.8.0 REQUIRED)
 message(STATUS "HDF5: found ${HDF5_LIBRARIES}")
endif()
include_directories(${HDF5_INCLUDE_DIR})
add_definitions(${HDF5_CPPFLAGS})
add_definitions(-DWITH_HDF5)



#find_package(HDF5 1.8.0)

##############################################################################
# Gurobi
##############################################################################
find_package(GUROBI)
if(GUROBI_FOUND)
    include_directories(${GUROBI_INCLUDE_DIR})
endif()

##############################################################################
# C++11 support
##############################################################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Some functionality will not be available.")
    ADD_DEFINITIONS(-DNO_CXX11)
endif()


##############################################################################
# Boost Python vigra vigra support
##############################################################################
OPTION(BUILD_PYTHON "build python bindings" ON)


if(BUILD_PYTHON)
    #--------------------------------------------------------------
    # vigra
    #--------------------------------------------------------------
    FIND_PACKAGE(VIGRA REQUIRED)
    include_directories(${VIGRA_INCLUDE_DIR})

    #--------------------------------------------------------------
    # Boost
    #--------------------------------------------------------------
    find_package(Boost 1.47 REQUIRED)
    find_package(Boost COMPONENTS python REQUIRED)

    if((Boost_PYTHON_LIBRARY MATCHES "boost_python\\.lib") OR
       (Boost_PYTHON_LIBRARY MATCHES "boost_python3\\.lib"))
        ADD_DEFINITIONS(-DBOOST_AUTO_LINK_NOMANGLE)
    endif()
    #--------------------------------------------------------------
    # Python Lib
    #--------------------------------------------------------------
    find_package (PythonLibs REQUIRED)
    include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${Boost_INCLUDE_DIR}
        ${Boost_PYTHON_INCLUDE_DIR}
    )


    add_library( _graph SHARED
        src/andres/graph/python/graph_module.cxx
        src/andres/graph/python/grid_graph.cxx
        src/andres/graph/python/graph.cxx
        src/andres/graph/python/lifted_mc_model.cxx
        src/andres/graph/python/lifted_mc.cxx
        src/andres/graph/python/learn_lifted.cxx
        # lifted mc solvers
        src/andres/graph/python/lifted_mc_ga_solver.cxx
        src/andres/graph/python/lifted_mc_kl_solver.cxx
        src/andres/graph/python/lifted_mc_parallel_solver.cxx
    )
    set_target_properties(_graph PROPERTIES PREFIX "" SUFFIX ".so")

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/graph)

    add_custom_target(copy_python__init__ ALL#POST_BUILD
        COMMAND cmake -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/andres/graph/python/module/__init__.py
        ${CMAKE_BINARY_DIR}/python/graph/__init__.py)

    ADD_CUSTOM_COMMAND(
            TARGET _graph
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_graph> ${CMAKE_BINARY_DIR}/python/graph/
            COMMENT "Copying target _graph to temporary module directory")

    #DEPENDS _graph )

    #add_custom_target(copy_python_test POST_BUILD
    #COMMAND cmake -E copy_if_different ${CMAKE_SOURCE_DIR}/test_graph.py ${CMAKE_BINARY_DIR}/test_graph.py
    #DEPENDS _graph )

    #--------------------------------------------------------------
    # Link libraries
    #--------------------------------------------------------------
    find_library(RT rt)
    target_link_libraries( _graph
        ${PYTHON_LIBRARIES}
        ${Boost_PYTHON_LIBRARIES}
        ${VIGRA_IMPEX_LIBRARY}
        ${HDF5_LIBRARIES}
    )


endif()

#if(DOXYGEN_FOUND)
#    configure_file("${graph_SOURCE_DIR}/doxygen/doxyfile-graph.in" "${graph_BINARY_DIR}/doxyfile-graph" @ONLY IMMEDIATE)
#    add_custom_target(doc-graph ALL COMMAND ${DOXYGEN} "${graph_BINARY_DIR}/doxyfile-graph")
#endif()
