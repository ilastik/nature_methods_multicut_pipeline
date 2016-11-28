// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL andres_graph_PyArray_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <andres/graph/hdf5/graph.hxx>
#include <andres/graph/hdf5/grid-graph.hxx>
#include <vector>
#include <string>


#include "andres/graph/grid-graph.hxx"
#include "andres/functional.hxx"
#include "andres/graph/graph.hxx"
#include "andres/graph/threadpool.hxx"
#include "andres/graph/neighborhood.hxx"
#include "andres/graph/multicut-lifted/lifted_mc_model.hxx"

#include "image.hxx"

namespace bp = boost::python;
namespace agraph = andres::graph;





template<class MODEL>
void addLongRangeEdges(
    MODEL & model,
    vigra::NumpyArray<2, float> edgePmap, // in 0-1
    const float beta = 0.5f,
    const int minRadius = 2,
    const int maxRadius = 7                                                
){  
    GRAPH_CHECK_OP(model.originalGraph().shape(0), == ,edgePmap.shape(0),"");
    GRAPH_CHECK_OP(model.originalGraph().shape(1), == ,edgePmap.shape(1),"");

    typedef vigra::TinyVector<int,   2> Coord;
    const auto shape = edgePmap.shape();

    auto clipToImg = [&](const Coord & coord){
        auto c = coord;
        for(auto i=0; i<2; ++i){
            c[i] = std::max(0, c[i]);
            c[i] = std::min(int(shape[i]),c[i]);
        }
        return c;
    };


    // move a coordinate to local min
    auto moveToMin = [&](const Coord & coord){
        Coord coords[5] = {
                coord,
                clipToImg(coord+Coord(0,1)),
                clipToImg(coord+Coord(1,0)),
                clipToImg(coord+Coord(-1,-1)),
                clipToImg(coord+Coord(-1, 0))
        };
        auto minVal = std::numeric_limits<float>::infinity();
        auto minCoord = Coord();
        for(size_t i=0; i<5; ++i){
            const auto val = edgePmap[coord];
            if(val<minVal){
                minVal = val;
                minCoord = coord;
            }
        }
        return minCoord;
    };
    
    auto & originalGraph = model.originalGraph();
    auto & liftedGraph = model.liftedGraph();






    int rad = 5;
    
    auto pOpt = agraph::ParallelOptions();
    const auto nThreads = pOpt.numThreads(-1).getActualNumThreads();
    
    typedef vigra::TinyVector<float, 2> FCoord;
    auto node = [&](const Coord & coord){
        return size_t(coord[0] + coord[1]*shape[0]);
    };

    std::mutex graphMutex;
    std::mutex mapMutex;

    std::set<size_t > processed;

    struct ToAdd{
        size_t u,v;
        float w;
    };

    size_t bufferSize = 1000000;
    std::vector<std::vector<ToAdd> > buffers(nThreads);
    for(auto & vec : buffers){
        vec.reserve(bufferSize+1);
    }



    auto addToBuffer = [&](std::vector<ToAdd> & buffer,const size_t u_, const size_t v_, const float w_){

        ToAdd ta;
        ta.u=u_;
        ta.v=v_;
        ta.w=w_;
        buffer.push_back(ta);
        //std::cout<<"buffer size"<<buffer.size()<<"\n";
        if(buffer.size()>=bufferSize){

            std::unique_lock<std::mutex> lock(graphMutex);
            //std::cout<<"clear buffer\n";
            for(const auto & ta  : buffer){
                const auto fe = liftedGraph.findEdge(ta.u,ta.v);
                // not yet  in lifted graph
                // therefore cannot be processed
                if(!fe.first){
                    //std::cout<<"a\n";
                    const auto e = model.setCost(ta.u,ta.v,ta.w,false);
                    processed.insert(e);
                }
                // edge is in lifted graph
                else{
                    //std::cout<<"b\n";
                    auto fm = processed.find(fe.second);
                    // not yet processed
                    if(fm == processed.end()){
                        std::cout<<"b1\n";
                        const auto e = model.setCost(ta.u,ta.v,ta.w,false);
                        processed.insert(e);
                    }
                    else{
                        std::cout<<"b2\n";
                    }
                }
            }
            buffer.resize(0);
        }
    };


    std::cout<<"lifted graph edge num "<<liftedGraph.numberOfEdges()<<"\n";
    agraph::parallel_foreach(
        nThreads,
        shape[1],
        [&](const int threadId, const int y){
            auto & buffer = buffers[threadId];
            for(int x=0; x<shape[0]; ++x){
                const auto p = Coord(x,y);
                const auto u = node(p);
                GRAPH_CHECK_OP(u,<,originalGraph.numberOfVertices(),"");
                GRAPH_CHECK_OP(u,<,liftedGraph.numberOfVertices(),"");
                const auto start = clipToImg(p-maxRadius);
                const auto end = clipToImg(p+maxRadius+1);
                auto q = Coord();
                for(q[0]=start[0]; q[0]<end[0]; ++q[0])
                for(q[1]=start[1]; q[1]<end[1]; ++q[1]){

                    GRAPH_CHECK_OP(q[0],>=,0,"");
                    GRAPH_CHECK_OP(q[1],>=,0,"");
                    GRAPH_CHECK_OP(q[0],<,shape[0],"");
                    GRAPH_CHECK_OP(q[1],<,shape[1],"");
                    const auto v = node(q);

                    size_t e;
                    if( norm(p-q) < float(minRadius))
                        continue;
                    if(p==q || v>u){
                        continue;
                    }

                    
                    GRAPH_CHECK_OP(v,<,originalGraph.numberOfVertices(),"");
                    GRAPH_CHECK_OP(v,<,liftedGraph.numberOfVertices(),"");
                    const auto qf = FCoord(q);
                    const auto pq = q-p;
                    const auto dist = vigra::norm(pq);
                    const auto step =  pq*(1.0f / (dist * 1.3f + 0.5f));
                    auto pOnLine = FCoord(p);
                    auto noMax = true;
                    auto maxVal = -1.0f*std::numeric_limits<float>::infinity();
                    while(Coord(pOnLine)!=q){
                        //std::cout<<"pol "<<pOnLine<<"\n";
                        auto iCord = Coord(pOnLine);
                        if(iCord != p){
                            noMax = false;
                            maxVal = std::max(edgePmap[iCord], maxVal);
                        }
                        pOnLine += step;
                    }
                    const double p1 = std::max(std::min(maxVal,0.999f),0.001f);
                    const double p0 = 1.0 - p1;
                    auto w = std::log(p0/p1) + beta;

                    addToBuffer(buffer, u,v,w);
                }
            }
        }
    );

    // clear whats left in buffers
    for(const auto & buffer : buffers){
        for(const auto & ta : buffer){
            const auto fe = liftedGraph.findEdge(ta.u,ta.v);
            // not yet  in lifted graph
            // therefore cannot be processed
            if(!fe.first){
                const auto e = model.setCost(ta.u,ta.v,ta.w,false);
                processed.insert(e);
            }
            // edge is in lifted graph
            else{
                auto fm = processed.find(fe.second);
                // not yet processed
                if(fm == processed.end()){
                    const auto e = model.setCost(ta.u,ta.v,ta.w,false);
                    processed.insert(e);
                }
            }
        }
    }
    std::cout<<"lifted graph edge num "<<liftedGraph.numberOfEdges()<<"\n";
}




template<class MODEL>
void addLongRangeNH(
    MODEL & model,
    const int radius = 7                                                
){  
    for(auto u=0; u<model.originalGraph().numberOfVertices(); ++u){
        const auto otherNodes = agraph::verticesInGraphNeigborhood(model.originalGraph(), u, radius);
        for(const auto v : otherNodes){
            if(u!=v){
                model.setCost(u,v,0.0);
            }
        }
    }
}



template<class LiftedMcModel>
double evalCut(
    LiftedMcModel & liftedMcModel,
    vigra::NumpyArray<1, uint8_t> cut
){
    auto s = 0.0;
    for(auto e=0; e<liftedMcModel.liftedGraph().numberOfEdges(); ++e){
        if(cut[e]){
            s += liftedMcModel.edgeCosts()[e];
        }
    }
    return s;
}




template<class LiftedMcModel>
void setCosts(
    LiftedMcModel & liftedMcModel,
    vigra::NumpyArray<1, vigra::TinyVector<uint64_t, 2> > uv,
    vigra::NumpyArray<1, float> costs,
    const bool overwrite 
){
    GRAPH_CHECK_OP(uv.size(), == , costs.size(), "shape mismatch: uv and costs have different size");

    const auto u = uv.bindElementChannel(0);
    const auto v = uv.bindElementChannel(1);
    const auto c = vigra::MultiArrayView<1, float>(costs);
    liftedMcModel.setCosts(u.begin(), u.end(), v.begin(), c.begin(), overwrite);
}


template<class LiftedMcModel>
vigra::NumpyAnyArray edgeLabelsToNodeLabels(
    const LiftedMcModel & model,
    vigra::NumpyArray<1, uint8_t> edgeLabels,
    vigra::NumpyArray<1, uint64_t> nodeLabels
){
    vigra::TinyVector<int,1> shape(model.originalGraph().numberOfVertices());
    nodeLabels.reshapeIfEmpty(shape);
    model.getNodeLabels(edgeLabels, nodeLabels);
    return nodeLabels;
}


template<class LiftedMcModel>
vigra::NumpyAnyArray nodeLabelsToEdgeLabels(
    const LiftedMcModel & model,
    vigra::NumpyArray<1, uint64_t> nodeLabels,
    vigra::NumpyArray<1, uint8_t> edgeLabels
){
    vigra::TinyVector<int,1> shape(model.liftedGraph().numberOfEdges());
    edgeLabels.reshapeIfEmpty(shape);
    model.getEdgeLabels(nodeLabels, edgeLabels);
    return edgeLabels;
}


template<class LiftedMcModel>
void fuseGtObjective(
    LiftedMcModel & model,
    vigra::NumpyArray<2, uint64_t> nodeLabels,
    const size_t rr,
    const double beta,
    const bool verbose
){
    if(verbose)
        std::cout<<"nodeLabel.shape "<<nodeLabels.shape()<<"\n";

   
    const auto & originalGraph = model.originalGraph();
    const auto & liftedGraph = model.liftedGraph();
    auto nV = originalGraph.numberOfVertices();
    auto nGt = nodeLabels.shape(1);

    std::vector<std::set<size_t> > extendedNh(nV);
    std::vector<std::set<size_t> > extendedNh2(nV);


    for(auto n=0; n<originalGraph.numberOfVertices(); ++n){
        for(auto iter = originalGraph.verticesFromVertexBegin(n); iter!=originalGraph.verticesFromVertexEnd(n); ++iter){
            extendedNh[n].insert(*iter);
            extendedNh2[n].insert(*iter);
        }
    }

    for(auto r=0;r<2;++r){

        for(auto n=0; n<originalGraph.numberOfVertices(); ++n){
            auto & thisNodeNhSet = extendedNh2[n];
            for(auto iter = originalGraph.verticesFromVertexBegin(n); iter!=originalGraph.verticesFromVertexEnd(n); ++iter){
                auto otherNode = *iter;
                const auto & otherNodeNhSet = extendedNh[otherNode];
                thisNodeNhSet.insert(otherNodeNhSet.begin(), otherNodeNhSet.end());
                thisNodeNhSet.erase(n);
            }
        }
        extendedNh = extendedNh2;
    }

    for(auto u=0; u<originalGraph.numberOfVertices(); ++u){
        const auto & enh = extendedNh[u];
        //std::cout<<"node "<<u<<" |enh| = "<<enh.size()<<"\n";

        const auto uGt = nodeLabels.bindInner(u); 
        for(const auto v : enh){
            const auto vGt = nodeLabels.bindInner(v); 
            auto p1 = 0.0;
            for(size_t gtc=0; gtc<nGt; ++gtc){
                p1 += uGt[gtc] != vGt[gtc] ?  1 : 0;
            }
            p1 /= nGt;
            //if (p1>0.1)
            //    std::cout<<"   p1 "<<p1<<"\n";

            p1 = std::max(std::min(0.999, p1),0.001);
            auto p0 = 1.0 - p1;
            auto w = std::log(p0/p1) + std::log((1.0-beta)/beta);
            model.setCost(u,v,w);
        }
    }
}



template<class LiftedMcModel>
void fuseGtObjectiveGrid(
    LiftedMcModel & model,
    vigra::NumpyArray<3, uint64_t> nodeLabels,
    vigra::NumpyArray<1, double>    pExpert,
    vigra::NumpyArray<2, float>     cutRegularizer,
    const size_t rr,
    const double beta,
    //const double cLocal,
    const bool verbose
){
    if(verbose)
        std::cout<<"nodeLabel.shape "<<nodeLabels.shape()<<"\n";

   
    const auto & originalGraph = model.originalGraph();
    const auto & liftedGraph = model.liftedGraph();
    auto nV = originalGraph.numberOfVertices();
    typedef vigra::TinyVector<int,2> Coord;
    Coord shape(originalGraph.shape(0), originalGraph.shape(1));

    auto nGt = nodeLabels.shape(2);

    std::vector<std::set<size_t> > extendedNh(nV);
    std::vector<std::set<size_t> > extendedNh2(nV);

    auto clipToImg = [&](const Coord & coord){
        auto c = coord;
        for(auto i=0; i<2; ++i){
            c[i] = std::max(0, c[i]);
            c[i] = std::min(int(shape[i]),c[i]);
        }
        return c;
    };
    auto node = [&](const Coord & coord){
        return size_t(coord[0] + coord[1]*shape[0]);
    };

    Coord p;

    for(p[1]=0; p[1]<shape[1]; ++p[1])
    for(p[0]=0; p[0]<shape[0]; ++p[0]){

        const auto u = node(p);
        const auto start = clipToImg(p-int(rr));
        const auto end = clipToImg(p+int(rr)+1);

        auto lp = nodeLabels.bindInner(p[0]).bindInner(p[1]);
        auto q = Coord();

        for(q[1]=start[1]; q[1]<end[1]; ++q[1])
        for(q[0]=start[0]; q[0]<end[0]; ++q[0]){
            const auto v = node(q);
            if(p!=q && u < v){
                auto d = vigra::norm(p-q);
                if(d<=float(rr)){
                    if(d<=1.5)
                        d*=0.5;
                    auto lq = nodeLabels.bindInner(q[0]).bindInner(q[1]);
                    auto p0 = 0.0;
                    auto p1 = 0.0;
                    for(size_t gtc=0; gtc<nGt; ++gtc){
                        p0 += (lp[gtc] == lq[gtc]) ?  pExpert(gtc) : 0.0;
                        p1 += (lp[gtc] != lq[gtc]) ?  pExpert(gtc) : 0.0;
                    }
                    auto Z = p0+p1;
                    p0/=Z;
                    p1/=Z;
                    //if (p1>0.1)
                    //    std::cout<<"   p1 "<<p1<<"\n";

                    p1 = std::max(std::min(0.99999, p1),0.00001);
                    p0 = 1.0 - p1;
                    auto w = std::log(p0/p1) + std::log((1.0-beta)/beta);
                    if(d<=1.01){
                        auto c = (cutRegularizer[p] + cutRegularizer[q])/2.0;
                        w += c;
                    }
                    model.setCost(u,v,w*d);
                }
            }
        }
    }
}

template<class LiftedMcModel>
void thinObjectSeededSeg(
    LiftedMcModel & model,
    vigra::NumpyArray<2, uint64_t>  linkConstraints,
    vigra::NumpyArray<2, float>     cutCosts,
    const double constraintCost,
    const bool verbose
){

    auto shape = linkConstraints.shape();
    auto tshape = cutCosts.shape();
    GRAPH_CHECK_OP(shape[0],==,model.originalGraph().shape(0),"");
    GRAPH_CHECK_OP(shape[1],==,model.originalGraph().shape(1),"");

    GRAPH_CHECK_OP(shape[0]*2-1,==,tshape[0],"");
    GRAPH_CHECK_OP(shape[1]*2-1,==,tshape[1],"");

    typedef vigra::TinyVector<int,2> Coord;
    ///  link constraints
    std::map<uint64_t, std::vector<size_t> >   constraintSet;

    auto node = [&](const Coord & c){
        return c[0] + c[1]*shape[0];
    };
    std::cout<<"local terms\n";
    for(auto y=0; y<shape[1]; ++y)
    for(auto x=0; x<shape[0]; ++x){
        const Coord c(x,y);
        auto u = node(c);
        if(linkConstraints(x,y)!=255){
            constraintSet[linkConstraints(x,y)].push_back(u);
        }
        if(x+1<shape[0]){
            const Coord c2(x+1,y);
            const auto v = node(c2);
            auto val = cutCosts[c+c2];
            model.setCost(u,v,val);
        }
        if(y+1<shape[1]){
            const Coord c2(x,y+1);
            const auto v = node(c2);
            auto val = cutCosts[c+c2];
            model.setCost(u,v,val);
        }
    }
    std::cout<<"to vecvec\n";
    std::vector< std::vector<size_t> >   cVecVec;
    for(auto iter = constraintSet.begin(); iter!=constraintSet.end(); ++iter){
        cVecVec.push_back(iter->second);
    }
    std::cout<<"must link terms\n";
    // must link constraints
    for(auto & inodes : cVecVec){
        GRAPH_CHECK_OP(inodes.size(),>=,1,"");
        std::cout<<"inodes "<<inodes.size()<<"\n";
        if(inodes.size() >1){
            for(auto i=0; i<inodes.size()-1; ++i)
            for(auto j=i+1; j<inodes.size(); ++j){
                model.setCost(inodes[i],inodes[j], constraintCost);
            }
        }
    }
    std::cout<<"cannot link terms\n";
    // cannot link constraints
    for(auto i=0; i<cVecVec.size()-1; ++i){
        const auto & nodesI = cVecVec[i];
        for(auto j=i+1; j<cVecVec.size(); ++j){
            const auto & nodesJ = cVecVec[j];

            for(auto in : nodesI)
            for(auto jn : nodesJ){
                model.setCost(in,jn,-1.0*constraintCost);
            }
        }
    }

}



template<class LiftedMcModel>
void lmSuperpixel(
    LiftedMcModel & model,
    vigra::NumpyArray<2, float>     cutCosts,
    const size_t seedRadius = 10,
    const size_t stepSize = 2,
    const double sigma = 3.0,
    const double seedRepulsion = 1.0
){


    const int shape[2] = {
        int(model.originalGraph().shape(0)),
        int(model.originalGraph().shape(1)) 
    };
    auto tshape = cutCosts.shape();


    typedef vigra::TinyVector<int,2> Coord;

    auto node = [&](const Coord & c){
        return c[0] + c[1]*shape[0];
    };
    std::cout<<"generate seeds\n";


    std::vector<Coord> seeds;

    for(auto y=seedRadius+1; y<shape[1]-seedRadius-1; y += seedRadius)
    for(auto x=seedRadius+1; x<shape[0]-seedRadius-1; x += seedRadius){
        const Coord c(x,y);
        seeds.push_back(c);
    }

    std::cout<<"add repulsion number of seeds "<<seeds.size()<<"\n";
    GRAPH_CHECK_OP(seeds.size(),>=,2,"seedRadius is to large");
    // seed repulsion
    auto c=0;
    auto r=0;
    for(auto si=0; si<seeds.size()-1; ++si)
    for(auto sj=si+1; sj<seeds.size(); ++sj){
        const auto & ci = seeds[si];
        const auto & cj = seeds[sj];
        if(vigra::norm(ci-cj)<5*seedRadius){
            model.setCost(node(ci),node(cj),-1.0*seedRepulsion*seedRadius*100.0);
            ++c;
        }
        else{
            ++r;
        }
    }
    std::cout<<"repulsive edges "<<c<<" rejected "<<r<<"\n";

    std::cout<<"add seed attractiveness \n";

    auto nd = [&](const double dist){
        return (1.0/(sigma*std::sqrt(2.0*3.1415926))) *std::exp(-0.5*std::pow(dist/sigma,2));
    };
    for(auto & uCoord  : seeds){
        Coord vCoord;
        for(vCoord[0] = uCoord[0]-seedRadius; vCoord[0]<uCoord[0]+seedRadius+1;++vCoord[0])
        for(vCoord[1] = uCoord[1]-seedRadius; vCoord[1]<uCoord[1]+seedRadius+1;++vCoord[1]){
            auto  d = vigra::norm(uCoord-vCoord);
            auto  w = nd(d);
            model.setCost(node(uCoord), node(vCoord), w*20.0);
        }
    }

    std::cout<<"add grid graph cost attractiveness \n";
    std::cout<<"local terms\n";
    for(auto y=0; y<shape[1]; ++y)
    for(auto x=0; x<shape[0]; ++x){
        const Coord c(x,y);
        auto u = node(c);
        if(x+1<shape[0]){
            const Coord c2(x+1,y);
            const auto v = node(c2);
            auto val = cutCosts[c+c2];
            model.setCost(u,v,val);
        }
        if(y+1<shape[1]){
            const Coord c2(x,y+1);
            const auto v = node(c2);
            auto val = cutCosts[c+c2];
            model.setCost(u,v,val);
        }
    }
    std::cout<<"build model done\n";
}



template<class LiftedMcModel>
void andresImageSegModel(
    LiftedMcModel & model,
    vigra::NumpyArray<2, float> pmapImage,
    const float bias = 0.5
){
    const auto & shape = pmapImage.shape();
    // convert to andres image
    Image image(shape.begin(),shape.begin());
    for(auto y=0; y<shape[1]; ++y)
    for(auto x=0; x<shape[0]; ++x){
        image(x,y) = pmapImage(x,y);
    }

    const auto & graph = model.originalGraph();

    // boundary-probability-to-multicut-problem-image.cxx
    std::vector<float> edgeCutProbabilities;
    PixelCutProbabilityFromEdgesWTA<float, size_t> pixelCutProbabilityWTA(image);
    constructGraphAndEdgeProbabilities(pixelCutProbabilityWTA, graph, edgeCutProbabilities, bias);

}


template<class LiftedMcModel>
void loadBsdModel(LiftedMcModel & model, const std::string & filename){
    auto fileHandle = agraph::hdf5::openFile(filename);
    agraph::hdf5::load(fileHandle,"graph",model._originalGraph());
    agraph::hdf5::load(fileHandle,"graph-lifted",model._liftedGraph());
    std::vector<size_t> shape;
    auto & ec = model._edgeCosts();
    //ec.resize(model.liftedGraph().numberOfEdges());
    std::vector<double > ecd;
    agraph::hdf5::load(fileHandle, "edge-cut-probabilities", shape, ecd);

    transform(
       ecd.begin(),
       ecd.end(),
       ecd.begin(),
       andres::NegativeLogProbabilityRatio<double,double>()
    );

    ec.resize(ecd.size());
    std::copy(ecd.begin(), ecd.end(),ec.begin());
    agraph::hdf5::closeFile(fileHandle);


    transform(
       ec.begin(),
       ec.end(),
       ec.begin(),
       andres::NegativeLogProbabilityRatio<double,double>()
    );

}

template<class OG, class F>
void exportLiftedMcModelT(const std::string & clsName, F && f){
    typedef OG originalGraph;
    typedef agraph::multicut_lifted::LiftedMcModel<originalGraph, float> LiftedMcModel;


    auto cls = bp::class_<LiftedMcModel>
    (
        clsName.c_str(), 
        bp::init<
            const originalGraph&
        >(
            bp::arg("originalGraph")
        )[bp::with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const originalGraph & */>()]
    )
        .def("originalGraph",&LiftedMcModel::originalGraph , bp::return_internal_reference<>())
        .def("liftedGraph",&LiftedMcModel::liftedGraph , bp::return_internal_reference<>())
        .def("_setCosts",vigra::registerConverters(&setCosts<LiftedMcModel>))
        .def("_setCost",&LiftedMcModel::setCost)
        .def("evalCut",vigra::registerConverters(&evalCut<LiftedMcModel>))
        .def("edgeLabelsToNodeLabels",vigra::registerConverters(&edgeLabelsToNodeLabels<LiftedMcModel>),
            (
                bp::arg("edgeLabels"),
                bp::arg("out") = bp::object()
            )
        )
        .def("loadBsdModel", &loadBsdModel<LiftedMcModel>)
        .def("nodeLabelsToEdgeLabels",vigra::registerConverters(&nodeLabelsToEdgeLabels<LiftedMcModel>),
            (
                bp::arg("nodeLabels"),
                bp::arg("out") = bp::object()
            )
        );

    f(cls);


    bp::def("fuseGtObjective",vigra::registerConverters(&fuseGtObjective<LiftedMcModel>),
        (
            bp::arg("model"),
            bp::arg("nodeLabels"),
            bp::arg("rr") = 2,
            bp::arg("beta") = 0.5,
            bp::arg("verbose") = true
        )
    );
}


template<class MODEL>
vigra::NumpyAnyArray flattenLabels(
    const MODEL & model,
    vigra::NumpyArray<2, uint64_t> labels2d,
    vigra::NumpyArray<1, uint64_t> out
){
    vigra::TinyVector<int, 1> shape(labels2d.size());
    out.reshapeIfEmpty(shape);
    auto c=0;
    for(auto y=0; y<labels2d.shape(1); ++y)
    for(auto x=0; x<labels2d.shape(0); ++x){
        out[c] = labels2d(x,y);
        ++c;
    }
    return out;
}


void exportLiftedMcModel(){
    typedef agraph::GridGraph<2> GridGraph2D;
    typedef agraph::GridGraph<3> GridGraph3D;
    typedef agraph::Graph<> Graph;
    typedef agraph::multicut_lifted::LiftedMcModel<GridGraph2D, float> LiftedMcModelGridGraph2D;
    typedef agraph::multicut_lifted::LiftedMcModel<Graph, float> LiftedMcModelGraph;
    {
        typedef agraph::multicut_lifted::LiftedMcModel<GridGraph2D, float> LiftedMcModelGridGraph2D; 
        exportLiftedMcModelT<GridGraph2D>("LiftedMcModelGridGraph2D",
            [&](
                bp::class_< LiftedMcModelGridGraph2D > & cls
            ){
                cls
                    .def("flattenLabels",
                        vigra::registerConverters(&flattenLabels< LiftedMcModelGridGraph2D >),
                        (
                            bp::arg("labels"),
                            bp::arg("out") =  bp::object()
                        )
                    )
                ;
            }
        );
    }
    exportLiftedMcModelT<GridGraph3D>("LiftedMcModelGridGraph3D",
        [&](
            bp::class_<agraph::multicut_lifted::LiftedMcModel<GridGraph3D, float> > & cls
        ){
            
        }
    );
    exportLiftedMcModelT<Graph>("LiftedMcModelGraph",
        [&](
            bp::class_<agraph::multicut_lifted::LiftedMcModel<Graph, float> > & cls
        ){
            
        }
    );


    bp::def("addLongRangeEdges",vigra::registerConverters(&addLongRangeEdges<LiftedMcModelGridGraph2D>))
    ;

    bp::def("addLongRangeNH",vigra::registerConverters(&addLongRangeNH<LiftedMcModelGridGraph2D>))
    ;
    bp::def("addLongRangeNH",vigra::registerConverters(&addLongRangeNH<LiftedMcModelGraph>))
    ;

    bp::def("fuseGtObjectiveGrid",vigra::registerConverters(&fuseGtObjectiveGrid<LiftedMcModelGridGraph2D>),
        (
            bp::arg("model"),
            bp::arg("nodeLabels"),
            bp::arg("pExpert"),
            bp::arg("cutRegularizer"),
            bp::arg("rr") = 2,
            bp::arg("beta") = 0.5,
            //bp::arg("cLocal") = 100,
            bp::arg("verbose") = true
        )
    );

    bp::def("thinObjectSeededSeg",vigra::registerConverters(&thinObjectSeededSeg<LiftedMcModelGridGraph2D>),
        (
            bp::arg("model"),
            bp::arg("linkConstraints"),
            bp::arg("cutCosts"),
            bp::arg("c") = 1000.0,
            bp::arg("verbose") = true
        )
    );

    

    bp::def("lmSuperpixel",vigra::registerConverters(&lmSuperpixel<LiftedMcModelGridGraph2D>),
        (
            bp::arg("model"),
            bp::arg("cutCosts"),
            bp::arg("seedRadius") = 10,
            bp::arg("stepSize") = 2,
            bp::arg("sigma") = 3.0,
            bp::arg("seedRepulsion") =1.0
        )
    );


    bp::def("andresImageSegModel",vigra::registerConverters(&andresImageSegModel<LiftedMcModelGridGraph2D>),
        (
            bp::arg("model"),
            bp::arg("pmap"),
            bp::arg("bias") = 0.5
        )
    );
}
