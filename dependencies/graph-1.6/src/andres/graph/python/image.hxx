#pragma once
#ifndef TOOLS_IMAGE_HXX
#define TOOLS_IMAGE_HXX

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <array>
#include "andres/runtime-check.hxx"
#include <andres/marray.hxx>

typedef unsigned char GrayLevel;
typedef andres::Marray<GrayLevel> Image;


template<class T = double, class S = std::size_t>
struct PixelCutProbabilityFromEdgesWTA {
    typedef T value_type;
    typedef S size_type;


    PixelCutProbabilityFromEdgesWTA(const Image& img)
    :   image_(img){
        if(img.dimension() != 3 || img.shape(2) != 3) {
            throw std::runtime_error("not a color image.");
        }
    }

    //thats a dirty hack. edge probability image is assumed to have twice the size of the underlying image
    size_type shape(const size_type d) const { 
        return image_.shape(d)/2; 
    }

    value_type operator()(
        const size_type x0,
        const size_type y0,
        const size_type x1,
        const size_type y1
    ) const {

        assert(abs(x0-x1)+abs(y0-y1)==1);
        value_type valf=0;
        value_type eps=0.01; 
        value_type normalization=255;
        if(x1>x0){
            value_type first;
            value_type second;
            if(image_(2*x0,2*y0,0)+image_(2*x0,2*y0+1,0) > image_(2*x0+1,2*y0,0)+image_(2*x0+1,2*y0+1,0)){
                first = 0;
            }
            else{
                first = static_cast<value_type>(image_(2*x0+1,2*y0,0)+image_(2*x0+1,2*y0+1,0))/static_cast<value_type>((value_type(1)+eps)*2*normalization);
            }
            if (image_(2*x1,2*y1,0)+ image_(2*x1,2*y1+1,0)< image_(2*x1+1,2*y1,0)+ image_(2*x1+1,2*y1+1,0)){
                second=0;
            }
            else{
                second = static_cast<value_type>(image_(2*x1,2*y1,0)+image_(2*x1,2*y1+1,0))/static_cast<value_type>((value_type(1)+eps)*2*normalization);
            }
            valf=eps + std::max(first, second);
        }
        else{
            value_type first;
            value_type second;
            if(image_(2*x0,2*y0,0)+ image_(2*x0+1,2*y0,0)>image_(2*x0,2*y0+1,0)+ image_(2*x0+1,2*y0+1,0)){
                first = 0;
            }
            else{
                first = static_cast<value_type>(image_(2*x0,2*y0+1,0)+image_(2*x0+1,2*y0+1,0))/static_cast<value_type>((value_type(1)+eps)*2*normalization);
            }
            if(image_(2*x1,2*y1,0)+ image_(2*x1+1,2*y1,0)<image_(2*x1,2*y1+1,0)+ image_(2*x1+1,2*y1+1,0)){
                second = 0;
            }
            else{
                second = static_cast<value_type>(image_(2*x1,2*y1,0)+image_(2*x1+1,2*y1,0))/static_cast<value_type>((value_type(1)+eps)*2*normalization);
            }
            valf=eps + std::max(first, second);
        }
        return valf;
    }
    private:
    const Image& image_;
};

template<
    class PIXEL_CUT_PROBABILITY, 
    class VISITOR, 
    class T
>
void constructGraphAndEdgeProbabilities(
    const PIXEL_CUT_PROBABILITY& pixelCutProbability,
    const andres::graph::GridGraph<2, VISITOR>& graph,
    std::vector<T>& edgeCutProbabilities,
    const float bias
) {
    typedef PIXEL_CUT_PROBABILITY PixelDistance;
    typedef typename PixelDistance::value_type value_type;
    typedef typename PixelDistance::size_type size_type;
    typedef andres::graph::GridGraph<2, VISITOR> GridGraph;
    typedef typename GridGraph::EdgeCoordinate EdgeCoordinate;
    typedef typename GridGraph::VertexCoordinate VertexCoordinate;

    const size_type width = pixelCutProbability.shape(0);
    const size_type height = pixelCutProbability.shape(1);
    //logistic prior
    const value_type biasOnTheta = ::log(bias/(value_type(1)-bias));
    GRAPH_CHECK_OP(graph.shape(0),==,width,"");
    GRAPH_CHECK_OP(graph.shape(1),==,height,"");
    //graph.assign({{width,height}});
    edgeCutProbabilities.resize(graph.numberOfEdges());
    for(size_type y = 0; y < height; ++y) {
        for(size_type x = 0; x < width; ++x) {
            if(x < pixelCutProbability.shape(0) - 1) {
                const value_type distance = pixelCutProbability(x, y, x + 1, y);
                VertexCoordinate coord;
                coord[0] = x;
                coord[1] = y;
                const EdgeCoordinate ecRight(coord,0,false);
                const size_type e = graph.edge(ecRight);
		const value_type z = ::log(distance/(value_type(1) - distance));
                edgeCutProbabilities[e] = value_type(1)/(value_type(1)+::exp(-z-biasOnTheta));
            }
            if(y < pixelCutProbability.shape(1) - 1) {
                const value_type distance = pixelCutProbability(x, y, x, y + 1);
                VertexCoordinate coord;
                coord[0] = x;
                coord[1] = y;
                const EdgeCoordinate ecBelow(coord,1,false);
                const size_type e = graph.edge(ecBelow);
                const value_type z = ::log(distance/(value_type(1) - distance));
                edgeCutProbabilities[e] = value_type(1)/(value_type(1)+::exp(-z-biasOnTheta));
            }
        }
    }
}

#endif // #ifndef TOOLS_IMAGE_HXX
