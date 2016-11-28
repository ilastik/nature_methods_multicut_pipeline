// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL andres_graph_PyArray_API
#define NO_IMPORT_ARRAY

// On Mac, it's very important that Python.h is included before any C++ headers like <map>
#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/algorithm.hxx>

#include <map>

#include "andres/graph/grid-graph.hxx"
#include "andres/graph/graph.hxx"
#include "andres/graph/components.hxx"
#include "andres/graph/threadpool.hxx"
#include "andres/partition-comparison.hxx"
#include "andres/graph/multicut-lifted/lifted_mc_model.hxx"

#include <vigra/multi_array.hxx>


#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics.hpp>

namespace bp = boost::python;
namespace agraph = andres::graph;




template<class VALS_IN>
void stepFitterForRawData(
    VALS_IN & valsIn
){


    // try all steps
    const auto size = valsIn.size();
    if(size>=3){

        std::vector<uint8_t>  isLeft(size,false);
        isLeft[0] = true;

        auto evalFit = [&](){
            auto meanLeft = 0.0;
            auto meanRight = 0.0;
            auto cLeft = 0.0;
            auto cRight = 0.0;
            for(size_t i=0; i<size;++i){
                if(isLeft[i]){
                    meanLeft+=valsIn[i];
                    ++cLeft;
                }
                else{
                    meanRight+=valsIn[i];
                    ++cRight;
                }
            }
            meanLeft/=cLeft;
            meanRight/=cRight;
            auto d = 0.0;
            for(size_t i=0; i<size;++i){
                if(isLeft[i]){
                    d += std::pow(valsIn[i]-meanLeft,2);
                }
                else{
                    ++cLeft;
                    d += std::pow(valsIn[i]-meanRight,2);
                }
            }
            return std::make_tuple(d,meanLeft,meanRight,cLeft,cRight);
        };

        // find best fit
        auto bestI = 0;
        auto bestD = std::numeric_limits<double>::infinity();
        auto bestMeanLeft = 0.0;
        auto bestMeanRight = 0.0;
        auto bestCLeft = 0.0;
        auto besCRight = 0.0;
        for(size_t i=0; i<size-1; ++i){
            isLeft[i] = true;
            auto d = 0.0;
            auto meanLeft = 0.0;
            auto meanRight = 0.0;
            auto cLeft = 0.0;
            auto cRight = 0.0;
            std::tie(d,meanLeft,meanRight,bestCLeft,besCRight) = evalFit();
            if(d<bestD){
                bestI = i;
                bestD = d;
                bestMeanLeft = meanLeft;
                bestMeanRight = meanRight;
            }
        }
        bestD /= size;

 

        // size trust
        //  close to one mean we to trust this fit 
        //  a lot given the size
        auto sizeQ =  1.0 - std::exp(-0.005 * std::pow(float(10),2) );

        // features

        // transfer distance to a quality measure
        //  1 means perfect,  close to zero means poor
        auto fitQ = std::exp(0.05*bestD);       
        auto pfitQ = sizeQ*fitQ;
        auto sfitQ = sizeQ+fitQ;
        auto hfitQ = 2.0*pfitQ/sfitQ;
        auto stepHeight = std::abs(bestMeanLeft -  bestMeanRight);
        auto splitPos = (bestCLeft+1.0)/size;

    }

}

























template<unsigned int DIM, class F>
void forEachLineCoordinate(
    const vigra::TinyVector<int64_t, DIM>   shape,
    const vigra::TinyVector<float, DIM> start,
    const vigra::TinyVector<float, DIM> stop,  
    F && f                 
){
    typedef vigra::TinyVector<int64_t, DIM>  Coord;
    typedef vigra::TinyVector<float, DIM>  FCoord;

    auto clipToShape = [&](const Coord & coord){
        auto c = coord;
        for(auto d=0; d<DIM; ++d){
            c[d] = std::max(int64_t(0),c[d]);
            c[d] = std::min(shape[d]-1,c[d]);
        }
        return c;
    };

    auto startI = clipToShape(Coord(start));
    auto stopI = clipToShape(Coord(stop));

    auto diff = stopI - startI;
    auto d =  vigra::norm(diff);
    auto df = d*1.2+0.5;
    auto increment = diff/df;
    auto currentCoord = FCoord(startI);

    if(startI==stopI){
        f(startI);
    }
    else{
        Coord lastCoord(-1);
        while(clipToShape(Coord(currentCoord))!=stopI){
            if(clipToShape(Coord(currentCoord))!=lastCoord){
                f(clipToShape(Coord(currentCoord)));
                lastCoord = clipToShape(Coord(currentCoord));
            }
            currentCoord += increment;
        }
    }
}


template<unsigned int DIM>
void extractLineFeatures(
    vigra::MultiArrayView<DIM+1,float> pixelFeatures,
    const vigra::TinyVector<float, DIM>  start,
    const vigra::TinyVector<float, DIM>  stop,  
    vigra::MultiArrayView<1,float>  lineFeatures
){

    typedef vigra::TinyVector<int64_t, DIM+1>  PfCoord;
    typedef vigra::TinyVector<int64_t, DIM>  Coord;
    typedef vigra::TinyVector<float, DIM>  FCoord;

    const auto & pfShape = pixelFeatures.shape();
    Coord shape;
    std::copy(pfShape.begin(),pfShape.begin()+DIM,shape.begin());


    auto checkInShape = [&](const Coord & coord){
        auto c = coord;
        GRAPH_CHECK_OP(c[0],>=,0,"");
        GRAPH_CHECK_OP(c[1],>=,0,"");
        GRAPH_CHECK_OP(c[0],<=,shape[0],"");
        GRAPH_CHECK_OP(c[1],<=,shape[1],"");
        for(auto d=0; d<DIM; ++d){
            if(coord[d]<0 || coord[d]>=shape[d])
                return false;
        }
        return true;
    };

    std::vector<Coord> coords;
    forEachLineCoordinate<DIM>(shape,start,stop,
        [&](Coord lineCoord){
            GRAPH_CHECK(checkInShape(lineCoord),"");
            coords.push_back(lineCoord);
        }
    );
    GRAPH_CHECK_OP(coords.size(),>=,1,"");
    vigra::TinyVector<int64_t, 2> extractedLineShape(coords.size(),pixelFeatures.shape(DIM));
    vigra::MultiArray<2,float>  extractedLine(extractedLineShape);
    auto c = 0;
    for(const auto & coord : coords){
        PfCoord pfCoord;
        for(auto d=0; d<DIM;++d){
            pfCoord[d] = coord[d];
        }
        for(auto f=0; f<extractedLineShape[1]; ++f){
            pfCoord[DIM] = f;
            extractedLine(c,f) = pixelFeatures[pfCoord]; 
        }
        ++c;
    }

    using namespace boost::accumulators;


    auto replaceRotten = [](const float val, const float replaceVal){
        if(vigra::isfinite(val)){
            return val;
        }
        else{
            return replaceVal;
        }
    };
    
    // accumulate features from that
    auto outFeatIndex = 0;
    for(auto f=0; f<extractedLineShape[1]; ++f){

         std::size_t n = coords.size()*10; // number of MC steps
         std::size_t c =  1000; // cache size
       
        typedef accumulator_set<
            double, 
            stats<
                tag::mean,
                tag::min, 
                tag::max,
                tag::moment<2>,
                tag::moment<3>,
                tag::tail_quantile<boost::accumulators::right>
            > 
        > accumulator_t_left;
        //accumulator_set<double, stats<tag::mean, tag::moment<2> > > acc;


        //FreeChain a;
        if(coords.size()>=2){

            accumulator_t_left acc0( right_tail_cache_size = c );
            for(auto i=0; i<extractedLine.shape(0); ++i){
                auto isf = vigra::isfinite(extractedLine(i,f));
                GRAPH_CHECK(isf,"");
                acc0(double(extractedLine(i,f)));
            }
            auto mean = extract_result< tag::mean >(acc0);
            lineFeatures[outFeatIndex] = mean;                                                              ++outFeatIndex;
            lineFeatures[outFeatIndex] = mean*coords.size();                                                ++outFeatIndex;
            lineFeatures[outFeatIndex] = extract_result< tag::min >(acc0);                                  ++outFeatIndex;
            lineFeatures[outFeatIndex] = extract_result< tag::max >(acc0);                                  ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(extract_result< tag::moment<2> >(acc0),0.0);         ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(extract_result< tag::moment<3> >(acc0),0.0);         ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(quantile(acc0, quantile_probability = 0.1 ),mean);   ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(quantile(acc0, quantile_probability = 0.25 ),mean);  ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(quantile(acc0, quantile_probability = 0.5 ),mean);   ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(quantile(acc0, quantile_probability = 0.75 ),mean);  ++outFeatIndex;
            lineFeatures[outFeatIndex] = replaceRotten(quantile(acc0, quantile_probability = 0.90 ),mean);  ++outFeatIndex;
           //lineFeatures[outFeatIndex] = get<Mean>(a); ++outFeatIndex;
           //lineFeatures[outFeatIndex] = replaceRotten(get<StdDev>(a),0.0); ++outFeatIndex;
           //lineFeatures[outFeatIndex] = replaceRotten(get<Skewness>(a),0.0); ++outFeatIndex;
           //auto qs = get<Quantiles>(a);
           //for(auto q : qs){
           //    lineFeatures[outFeatIndex] = replaceRotten(q,get<Mean>(a)); ++outFeatIndex;
           //}   
        }
        else{
            // size is 1
            auto mean = extractedLine(0,f);
            lineFeatures[outFeatIndex] = mean;   ++outFeatIndex;
            lineFeatures[outFeatIndex] = mean;   ++outFeatIndex;
            lineFeatures[outFeatIndex] = mean;   ++outFeatIndex;
            lineFeatures[outFeatIndex] = mean;   ++outFeatIndex;
            lineFeatures[outFeatIndex] = 0;      ++outFeatIndex;
            lineFeatures[outFeatIndex] = 0;      ++outFeatIndex;
            lineFeatures[outFeatIndex] =mean;    ++outFeatIndex;
            lineFeatures[outFeatIndex] =mean;    ++outFeatIndex;
            lineFeatures[outFeatIndex] =mean;    ++outFeatIndex;
            lineFeatures[outFeatIndex] =mean;    ++outFeatIndex;
            lineFeatures[outFeatIndex] =mean;    ++outFeatIndex;
        }

    }
    GRAPH_CHECK_OP(outFeatIndex,==,lineFeatures.shape(0),"");
}


template<unsigned int DIM>
vigra::NumpyAnyArray lineFeatures(
    vigra::NumpyArray<1, vigra::TinyVector<uint64_t, 2> > uvIds,
    vigra::NumpyArray<1, vigra::TinyVector<float, DIM> > centers,
    vigra::NumpyArray<DIM+1, float>  pixelFeatures,
    vigra::NumpyArray<2, float> out
){

    typedef vigra::TinyVector<int64_t, DIM>  Coord;
    typedef vigra::TinyVector<float, DIM>  FCoord;

    const auto & pfShape = pixelFeatures.shape();
    Coord shape;
    std::copy(pfShape.begin(),pfShape.begin()+DIM,shape.begin());

    const auto nFeatIn = pixelFeatures.shape(DIM);
    const auto nFeatOut = 11*nFeatIn;

    vigra::TinyVector<int,2> lineFeaturesShape(uvIds.size(),nFeatOut);
    out.reshapeIfEmpty(lineFeaturesShape);   


    vigra::MultiArrayView<DIM+1,float> pixelFeaturesView(pixelFeatures);
    vigra::MultiArrayView<2,float> outView(out);
    auto e = 0;
    {
        vigra::PyAllowThreads _pythread;
        for(auto uv :uvIds){

            auto coordU = centers[uv[0]];
            auto coordV = centers[uv[1]];

            extractLineFeatures<DIM>(pixelFeaturesView, coordU, coordV, outView.bindInner(e));
            ++e;
        }
    }
    return out;
}


vigra::NumpyAnyArray newLineFeatures(
    vigra::NumpyArray<2, uint32_t> pixelLabels_,
    vigra::NumpyArray<3, float   > pixelFeatures_,
    vigra::NumpyArray<1, vigra::TinyVector<uint64_t, 2> > uvIds_,
    int nSamples 
){

    vigra::MultiArrayView<2, uint32_t> pixelLabels(pixelLabels_);
    vigra::MultiArrayView<3, float   > pixelFeatures(pixelFeatures_);
    vigra::MultiArrayView<1, vigra::TinyVector<uint64_t, 2> > uvIds(uvIds_);

    typedef vigra::TinyVector<int,2> Coord;
    typedef vigra::TinyVector<float,2> FCoord;
    // check input sanity
    auto minLabel = *std::min_element(pixelLabels.begin(),pixelLabels.end());
    GRAPH_CHECK_OP(pixelLabels.shape(0),==,pixelFeatures.shape(0),"");
    GRAPH_CHECK_OP(pixelLabels.shape(1),==,pixelFeatures.shape(1),"");
    GRAPH_CHECK_OP(minLabel,==,0,"");

    //std::cout<<"NUMBER OF INPUT FEATURES"<<pixelFeatures.shape()<<"\n";
    // some sizes
    auto shape = pixelLabels.shape();
    auto nEdges = uvIds.size();
    auto nNodes = *std::max_element(pixelLabels.begin(),pixelLabels.end()) + 1;

    // create output
    vigra::TinyVector<int,2> outShape(nEdges, pixelFeatures_.shape(2)*11+6);
    vigra::NumpyArray<2,float> out_;
    out_.reshapeIfEmpty(outShape);
    auto out = vigra::MultiArrayView<2,float>(out_);
    // allow threads from here on
    {


        auto minVal = *std::min_element(pixelFeatures.begin(), pixelFeatures.end());
        auto maxVal = *std::max_element(pixelFeatures.begin(), pixelFeatures.end());
        GRAPH_CHECK(vigra::isfinite(minVal),"");
        GRAPH_CHECK(vigra::isfinite(maxVal),"");



        vigra::PyAllowThreads allowThreads;

        std::vector<std::vector<Coord> > members(nNodes);
        std::vector<std::set<Coord> > borderMembersSet(nNodes);
        Coord c;
        Coord oc;
        //std::cout<<"members\n";
        for(c[1]=0; c[1]<shape[1]; ++c[1])
        for(c[0]=0; c[0]<shape[0]; ++c[0]){
            GRAPH_CHECK_OP(pixelLabels[c],<,nNodes,"");
            members[pixelLabels[c]].push_back(c);
            if(c[0]+1<shape[0]){
                oc = c;
                oc[0] +=1;
                if(pixelLabels[c]<pixelLabels[oc]){
                    borderMembersSet[pixelLabels[c]].insert(c);
                    borderMembersSet[pixelLabels[oc]].insert(oc);
                }
            }
            if(c[1]+1<shape[1]){
                oc = c;
                oc[1] +=1;
                if(pixelLabels[c]<pixelLabels[oc]){
                    borderMembersSet[pixelLabels[c]].insert(c);
                    borderMembersSet[pixelLabels[oc]].insert(oc);
                }
            }
        }
        //std::cout<<"fill vec\n";     
        std::vector<std::vector<Coord> > borderMembers(nNodes);
        agraph::parallel_foreach(-1,nNodes,
            [&](int threadId, int vertex){
                borderMembers[vertex].assign(borderMembersSet[vertex].begin(), borderMembersSet[vertex].end());
            }
        );
        //std::cout<<"the show\n";
        agraph::parallel_foreach(-1,nEdges,
            [&](int threadId, int edge){
                //std::cout<<"edge "<<edge<<"\n";
                const auto u = uvIds[edge][0];
                const auto v = uvIds[edge][1];
                // sanity check
                GRAPH_CHECK_OP(u,<,nNodes,"");
                GRAPH_CHECK_OP(v,<,nNodes,"");
                GRAPH_CHECK_OP(u,!=,v,"");
                const auto & mU = members[u];
                const auto & mV = members[v];
                const auto sizeU = mU.size();
                const auto sizeV = mV.size();
                // sanity check
                GRAPH_CHECK_OP(sizeU,>,0,"");
                GRAPH_CHECK_OP(sizeV,>,0,"");


                const auto & mbU = borderMembers[u];
                const auto & mbV = borderMembers[v];
                const auto sizeBU = mbU.size();
                const auto sizeBV = mbV.size();
                // sanity check
                GRAPH_CHECK_OP(sizeBU,>,0,"");
                GRAPH_CHECK_OP(sizeBV,>,0,"");

                auto featureIndex = 0;

                // start the show:
                // - compute pairwise distances
                //  and remember them for later use
                float minDist = std::numeric_limits<float>::infinity();
                float maxDist = -1.0*std::numeric_limits<float>::infinity();
                float meanDist = 0.0;
                auto nP = 0;
                auto minIndex = -1;
                std::vector<float> distVec;

                for(auto cv : mbV)
                for(auto cu : mbU){
                    const auto dist = float(vigra::norm(cu-cv));
                    meanDist += nP;
                    if(dist<minDist){
                        minDist = dist;
                        minIndex = nP;
                    }
                    maxDist = std::max(dist,maxDist);
                    distVec.push_back(dist);
                    ++nP;
                }
                GRAPH_CHECK_OP(nP,==,mbV.size()*mbU.size(),"");
                meanDist/=nP;


                // store features distance (6)
                out(edge, featureIndex) = minDist; ++featureIndex;
                out(edge, featureIndex) = maxDist; ++featureIndex;
                out(edge, featureIndex) = maxDist-minDist; ++featureIndex;
                out(edge, featureIndex) = meanDist; ++featureIndex;
                out(edge, featureIndex) = meanDist-minDist; ++featureIndex;
                out(edge, featureIndex) = meanDist-maxDist; ++featureIndex;


                //index sort the distance
                std::vector<int> indices(nP) ; 
                std::iota (std::begin(indices), std::end(indices), 0);
                vigra::indexSort(distVec.begin(),distVec.end(), indices.begin());

                // take the top n pairs
                // with the lowest distances
                
                vigra::TinyVector<int, 1> lineFeaturesShape(pixelFeatures.shape(2)*11);
                auto lineFeature = vigra::MultiArray<1, float>(lineFeaturesShape);
                auto lineFeatureMean = vigra::MultiArray<1, float>(lineFeaturesShape);

                auto nPairs = std::min(nP,nSamples);
                for(size_t pc=0; pc<nPairs; ++pc){

                    using namespace vigra::multi_math;
                    const auto pairIndex = indices[pc];
                    auto indexV = pairIndex / mbU.size();
                    auto indexU = pairIndex - indexV*mbU.size();

                    // sanity check
                    GRAPH_CHECK_OP(indexU,<,mbU.size(),"");
                    GRAPH_CHECK_OP(indexV,<,mbV.size(),"");

                    const auto & coordU = mbU[indexU];
                    const auto & coordV = mbV[indexV];
                    //std::cout<<coordU<<" "<<coordV<<"\n";
                    // compute the line features
                    vigra::MultiArrayView<1,float> lineFeaturesView(lineFeature);
                    extractLineFeatures<2>(pixelFeatures, FCoord(coordU), FCoord(coordV), lineFeaturesView);
                    lineFeatureMean += lineFeature;
                }
                lineFeatureMean /= nPairs;
                for(auto f : lineFeatureMean)
                    out(edge, featureIndex) = f; ++featureIndex;
            }
        );  
        minVal = *std::min_element(out.begin(), out.end());
        maxVal = *std::max_element(out.begin(), out.end());
        GRAPH_CHECK(vigra::isfinite(minVal),"");
        GRAPH_CHECK(vigra::isfinite(maxVal),"");


    }

    return out_;
}

/*
 * shortest path based on weights
 *
 */

vigra::NumpyAnyArray pathFeatures(
    const agraph::Graph<> &         originalGraph,
    const agraph::Graph<> &         liftedGraph,
    vigra::NumpyArray<2, float>     weights,
    vigra::NumpyArray<2, float>     edgeFeatures,
    vigra::NumpyArray<2, float>     nodeFeatures
){
    // FIXME: unfinished work-in-progress
    assert(false);
    return vigra::NumpyAnyArray();
#if 0
    // iterate in parallel over all nodes
    agraph::parallel_foreach(-1, originalGraph.numberOfVertices(),
        [&](int threadId, int u){

            // collect all the outgoing nodes
            std::vector<size_t> others;
            auto iter = liftedGraph.adjacenciesFromVertexBegin(u);
            auto endIter = liftedGraph.adjacenciesFromVertexEnd(u);
            for( ; iter!=endIter; ++iter){

                const auto & adj = *iter;
                auto v = adj.vertex();
                auto edge = adj.edge();

            }
        } 
    );
#endif
};

double evalRandIndex(
    vigra::NumpyArray<2, uint32_t> gt,
    vigra::NumpyArray<2, uint32_t> seg
){
    return andres::randIndex(gt.begin(),gt.end(),seg.begin(),true);
}

vigra::NumpyAnyArray makeResImg(
    vigra::NumpyArray<2, uint32_t> seg,
    vigra::NumpyArray<2, uint32_t> out
){
    out.reshapeIfEmpty(seg.shape());
    std::fill(out.begin(),out.end(),255);
    for(auto y=0; y<seg.shape(1); ++y)
    for(auto x=0; x<seg.shape(0); ++x){
        if(x+1<seg.shape(0)){
            if(seg(x,y)!=seg(x+1,y)){
                out(x,y) = 0;
                out(x+1,y) =0;
            }
        }
        if(y+1<seg.shape(1)){
            if(seg(x,y)!=seg(x,y+1)){
                out(x,y) = 0;
                out(x,y+1) =0;
            }
        }
    }
    return out;
}


template<unsigned int DIM>
vigra::NumpyAnyArray candidateSegToRagSeg(
    vigra::NumpyArray<DIM, uint32_t> ragLabels,
    vigra::NumpyArray<DIM, uint32_t> candidateLabels,
    vigra::NumpyArray<1, vigra::TinyVector<uint64_t, 2> > uvIds,
    vigra::NumpyArray<1, float> out
){  
    out.reshapeIfEmpty(uvIds.shape());

    {
        vigra::PyAllowThreads allowThreads;
        std::map<
            uint32_t,
            std::map<uint32_t, uint32_t>
        > overlapsWith;
        std::map<uint32_t, uint32_t> counters;

        //std::cout<<"ol mat\n";
        for(auto i=0; i<ragLabels.size(); ++i){
            const auto rl = ragLabels[i];
            const auto cl = candidateLabels[i]; 
            ++overlapsWith[rl][cl];
            ++counters[rl];
        }
        //std::cout<<"feat\n";
        for(auto e=0; e<uvIds.size(); ++e){
            const auto u = uvIds[e][0];
            const auto v = uvIds[e][1];
            const auto sU = float(counters[u]);
            const auto sV = float(counters[v]);
            const auto & olU = overlapsWith[u];
            const auto & olV = overlapsWith[v];

            auto isDiff = 0.0;
            auto cc = 0.0;
            for(const auto & keyAndSizeU : olU)
            for(const auto & keyAndSizeV : olV){

                auto keyU =  keyAndSizeU.first;
                auto rSizeU = float(keyAndSizeU.second)/sU;
                auto keyV =  keyAndSizeV.first;
                auto rSizeV = float(keyAndSizeV.second)/sV;

                if(keyU != keyV){
                    isDiff += (rSizeU * rSizeV);
                }
            }
            out[e] = isDiff;
        }
    }
    //std::cout<<"done\n";
    return out;
}


void exportLearnLifted(){
    typedef agraph::GridGraph<2> GridGraph2D;
    typedef agraph::GridGraph<3> GridGraph3D;
    typedef agraph::Graph<> Graph;
    typedef agraph::multicut_lifted::LiftedMcModel<GridGraph2D, float> LiftedMcModelGridGraph2D;
    typedef agraph::multicut_lifted::LiftedMcModel<GridGraph3D, float> LiftedMcModelGridGraph3D;
    typedef agraph::multicut_lifted::LiftedMcModel<Graph, float> LiftedMcModelGraph;


    bp::def("candidateSegToRagSeg",vigra::registerConverters(&candidateSegToRagSeg<2>),
        (
            bp::arg("ragLabels"),
            bp::arg("candidateLabels"),
            bp::arg("uvIds"),
            bp::arg("out") = bp::object()
        )
    );
    bp::def("candidateSegToRagSeg",vigra::registerConverters(&candidateSegToRagSeg<3>),
        (
            bp::arg("ragLabels"),
            bp::arg("candidateLabels"),
            bp::arg("uvIds"),
            bp::arg("out") = bp::object()
        )
    );


    bp::def("lineFeatures",vigra::registerConverters(&lineFeatures<2>),
        (
            bp::arg("uvIds"),
            bp::arg("centers"),
            bp::arg("pixelFeatures"),
            bp::arg("out") = bp::object()
        )
    )
    ;
    bp::def("pathFeatures", vigra::registerConverters(&pathFeatures),
        (
            bp::arg("originalGraph"),
            bp::arg("liftedGraph"),
            bp::arg("weights")
        )
    )
    ;
    bp::def("newLineFeatures", vigra::registerConverters(&newLineFeatures),
        (
            bp::arg("pixelLabels"),
            bp::arg("pixelFeatures"),
            bp::arg("uvIds"),
            bp::arg("nSamples")
        )
    )
    ;

    bp::def("evalRandIndex", vigra::registerConverters(&evalRandIndex),
        (
            bp::arg("gt"),
            bp::arg("seg")
        )
    )
    ;

    bp::def("makeResImg", vigra::registerConverters(&makeResImg),
        (
            bp::arg("seg"),
            bp::arg("out") = bp::object()
        )
    )
    ;
}
