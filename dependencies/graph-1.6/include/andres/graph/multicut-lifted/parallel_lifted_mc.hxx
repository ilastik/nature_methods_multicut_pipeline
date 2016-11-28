#pragma once
#ifndef ANDRES_GRAPH_MULTICUT_PARALLEL_LIFTED_MC_HXX
#define ANDRES_GRAPH_MULTICUT_PARALLEL_LIFTED_MC_HXX

#include <iomanip>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <stack>
#include <cmath>
#include <random>
#include <mutex>
#include <functional>
#include <set>


#include <vigra/resizeimage.hxx>
#include <vigra/convolution.hxx>

#include "andres/marray.hxx"
#include "andres/graph/grid-graph.hxx"
#include "andres/graph/threadpool.hxx"
#include "andres/graph/neighborhood.hxx"
#include "andres/graph/multicut-lifted/fusion_mover.hxx"
#include "andres/graph/multicut-lifted/greedy-additive.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"
#include "andres/graph/multicut-lifted/decomposing_lmc.hxx"


namespace andres {
namespace graph {
namespace multicut_lifted {


/*
    decompose grid into overlapping blocks
 
    solve each block on its on
 
    past beach block solution in a global solution
 
    fuse them via fusion moves
    
*/


class SolverSettings{
    enum Settings{
        KL = 0,
        GA = 1,
        FMR = 2
    };
};

template<class LIFTED_MC_MODEL>
class RandomizedProposalGenerator;

template<class LIFTED_MC_MODEL>
class RescaledProposalGenerator;

template<class LIFTED_MC_MODEL>
class SubgraphProposalGenerator;

template<class LIFTED_MC_MODEL, class PROPOSAL_GEN = SubgraphProposalGenerator<LIFTED_MC_MODEL> >
class ParallelSolver;

template<class LIFTED_MC_MODEL>
class SubgraphProposalGenerator{
public:

    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef SubgraphProposalGenerator<OTHER_LIFTED_MC_MODEL> type;
    };

    typedef GridGraph<2> GridGraph2D;
    typedef typename GridGraph2D::VertexCoordinate Coord2d;
    typedef std::vector<uint8_t>  LiftedEdgeLabels;
    struct  Settings{
        std::size_t subgraphRadius;
        int seed;
        Settings()
        :   subgraphRadius(40),
            seed(-1)
        {

        }
        template<class OTHER>
        Settings(const OTHER & other){
            this->subgraphRadius = other.subgraphRadius;
            this->seed = other.seed;
        }
    };

    SubgraphProposalGenerator(const LIFTED_MC_MODEL & model, const Settings & settings)
    :   model_(model),
        settings_(settings),
        rd_(),
        gen_(settings.seed == -1 ? std::mt19937(rd_()): std::mt19937(size_t(settings.seed)) ),
        //gen_(),
        nodeDist_(0,model.originalGraph().numberOfVertices()-1),
        edgeDist_(0,model.originalGraph().numberOfEdges()-1),
        globalToLocal_(model_.originalGraph().numberOfVertices()),
        localToGlobal_(model_.originalGraph().numberOfVertices())
    {

    }


    void generate(const LiftedEdgeLabels & currentBest, LiftedEdgeLabels & proposal){

        this->getSubgraph(model_.originalGraph(), currentBest, proposal );
    }

private:

    template<class SUB_MODEL>
    void optimizeSubmodel(const SUB_MODEL & subModel,
                          std::vector<uint8_t> & subgraphRes){

        // hard work is here
        std::vector<uint8_t> subgraphInput(subModel.liftedGraph().numberOfEdges(),1);
        subgraphRes.resize(subModel.liftedGraph().numberOfEdges());
        KernighanLinSettings s;
        s.verbose = false;
        kernighanLin(subModel.originalGraph(),subModel.liftedGraph(),
                     subModel.edgeCosts(),subgraphInput, subgraphRes,s);

        // transfer back to node solution
        

    }


    uint64_t getRandVar(const LiftedEdgeLabels & currentBest ){
        auto var = nodeDist_(gen_);
        for(auto i=0; i<1; ++i){
            auto edge = edgeDist_(gen_);
            if(currentBest[edge] == 1){
                var =  model_.originalGraph().vertexOfEdge(edge, 0);
                break;
            }
        }
        return var;
    }


    template<class G>
    void getSubgraph(const G & g, const LiftedEdgeLabels & currentBest, LiftedEdgeLabels & proposal){
    
        const size_t randVar = this->getRandVar(currentBest);
        const std::set<size_t> subgraphNodes = verticesInGraphNeigborhood(g,randVar,settings_.subgraphRadius);




        Graph<> subOg(subgraphNodes.size());

        //std::cout<<"OG  NODES "<<g.numberOfVertices()<<"\n";
        //::cout<<"SOG NODES "<<subOg.numberOfVertices()<<"\n";

        auto uuLocal = 0;
        for(auto uGlobal : subgraphNodes){
            localToGlobal_[uuLocal] = uGlobal;
            globalToLocal_[uGlobal] = uuLocal;
            ++uuLocal;
        }
        //std::cout<<"a\n";
        // fill normal graph
        for(auto uLocal =0; uLocal<subgraphNodes.size(); ++uLocal ){
            auto uGlobal = localToGlobal_[uLocal];
            auto aIter  = g.adjacenciesFromVertexBegin(uGlobal);
            auto aEnd  =  g.adjacenciesFromVertexEnd(uGlobal);
            while(aIter != aEnd){
                auto a = *aIter;
                auto e = a.edge();
                auto vGlobal = a.vertex();
                if(subgraphNodes.find(vGlobal) != subgraphNodes.end()){
                    auto vLocal = globalToLocal_[vGlobal];
                    GRAPH_CHECK_OP(vLocal,<,subgraphNodes.size(),"");
                    GRAPH_CHECK_OP(uLocal,<,subgraphNodes.size(),"");
                    subOg.insertEdge(uLocal, globalToLocal_[vGlobal]);
                }
                ++aIter;
            }
        }
        //std::cout<<"b\n";
        LiftedMcModel<Graph<>, float> subModel(subOg);
        // fill submodel
        for(auto uLocal =0; uLocal<subgraphNodes.size(); ++uLocal ){
            auto uGlobal = localToGlobal_[uLocal];
            auto aIter  = model_.liftedGraph().adjacenciesFromVertexBegin(uGlobal);
            auto aEnd  =  model_.liftedGraph().adjacenciesFromVertexEnd(uGlobal);
            while(aIter != aEnd){
                auto a = *aIter;
                auto e = a.edge();
                auto vGlobal = a.vertex();
                
                if(subgraphNodes.find(vGlobal) != subgraphNodes.end()){
                    auto vLocal = globalToLocal_[vGlobal];
                    GRAPH_CHECK_OP(uLocal,<,subgraphNodes.size(),"");
                    GRAPH_CHECK_OP(vLocal,<,subgraphNodes.size(),"");
                    subModel.setCost(uLocal, vLocal, model_.edgeCosts()[e]);
                }
                ++aIter;
            }
        }
        //std::cout<<"c\n";

        LiftedEdgeLabels subLabls;
        this->optimizeSubmodel(subModel, subLabls);

        //std::cout<<"c*\n";
        for(auto uLocal =0; uLocal<subgraphNodes.size(); ++uLocal ){
            auto uGlobal = localToGlobal_[uLocal];
            auto aIter  = model_.liftedGraph().adjacenciesFromVertexBegin(uGlobal);
            auto aEnd  =  model_.liftedGraph().adjacenciesFromVertexEnd(uGlobal);
            while(aIter != aEnd){
                auto a = *aIter;
                auto e = a.edge();
                auto vGlobal = a.vertex();
                
                if(subgraphNodes.find(vGlobal) != subgraphNodes.end()){
                    auto vLocal = globalToLocal_[vGlobal];
                    GRAPH_CHECK_OP(uLocal,<,subgraphNodes.size(),"");
                    GRAPH_CHECK_OP(vLocal,<,subgraphNodes.size(),"");
                    //subModel.setCost(uLocal, vLocal, model_.edgeCosts()[e]);

                    auto fres =  subModel.liftedGraph().findEdge(uLocal, vLocal);
                    proposal[e] =  subLabls[fres.second];

                }
                else{
                    proposal[e] = 1;
                }
                ++aIter;
            }
        }



    }

    void getSubgraph(const GridGraph<2> & g, const LiftedEdgeLabels & currentBest, LiftedEdgeLabels & proposal){
        Coord2d ggStart,subShape;   
        std::tie(ggStart, subShape) = this->getRandSubGridGraph(g,currentBest);
        GridGraph2D gg(subShape);

        LiftedMcModel<GridGraph2D, float> subModel(gg);

        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts  =model_.edgeCosts();

        auto uLocal = 0;
        for(auto ly=0; ly<subShape[1]; ++ly)
        for(auto lx=0; lx<subShape[0]; ++lx){
            auto ugx = lx + ggStart[0];
            auto ugy = ly + ggStart[1];
            auto uGlobal = ugx + ugy*g.shape(0);

            // all edges of global node where
            // both, u and v are in subgraph
            auto aIter  = liftedGraph.adjacenciesFromVertexBegin(uGlobal);
            auto aEnd  = liftedGraph.adjacenciesFromVertexEnd(uGlobal);
            while(aIter != aEnd){
                auto a = *aIter;
                auto e = a.edge();
                auto v = a.vertex();
                auto vgy = v/ g.shape(0);
                auto vgx = v - vgy*g.shape(1);
                if(vgx>=ggStart[0] && vgx <ggStart[0] + subShape[0] &&
                   vgy>=ggStart[1] && vgy <ggStart[1] + subShape[1]){
                    auto vlx = vgx - ggStart[0];
                    auto vly = vgy - ggStart[1];
                    auto vLocal  = vlx + vly*subShape[0];
                    subModel.setCost(uLocal,vLocal, edgeCosts[e]);
                }
                ++aIter;
            }
            ++uLocal;
        }
        LiftedEdgeLabels subLabls;
        this->optimizeSubmodel(subModel, subLabls);

        uLocal = 0;
        for(auto ly=0; ly<subShape[1]; ++ly)
        for(auto lx=0; lx<subShape[0]; ++lx){
            auto ugx = lx + ggStart[0];
            auto ugy = ly + ggStart[1];
            auto uGlobal = ugx + ugy*g.shape(0);

            // all edges of global node where
            // both, u and v are in subgraph
            auto aIter  = liftedGraph.adjacenciesFromVertexBegin(uGlobal);
            auto aEnd  = liftedGraph.adjacenciesFromVertexEnd(uGlobal);
            while(aIter != aEnd){
                auto a = *aIter;
                auto e = a.edge();
                auto v = a.vertex();
                auto vgy = v/ g.shape(0);
                auto vgx = v - vgy*g.shape(1);
                if(vgx>=ggStart[0] && vgx <ggStart[0] + subShape[0] &&
                   vgy>=ggStart[1] && vgy <ggStart[1] + subShape[1]){
                    auto vlx = vgx - ggStart[0];
                    auto vly = vgy - ggStart[1];
                    auto vLocal  = vlx + vly*subShape[0];
                    auto eLocal = subModel.liftedGraph().findEdge(vLocal,  uLocal).second;
                    proposal[e] = subLabls[eLocal];
                }
                else{
                    proposal[e] = 1;
                }
                ++aIter;
            }
            ++uLocal;
        }

    }


    std::pair<Coord2d, Coord2d>
    getRandSubGridGraph(const GridGraph<2> & g, const LiftedEdgeLabels & currentBest){

        const size_t randVar = this->getRandVar(currentBest);
        const int y = randVar / g.shape(0);
        const int x = randVar - y*g.shape(0);
        const int r = settings_.subgraphRadius;
        const int startX = std::max(0, x - r);
        const int startY = std::max(0, y - r);
        const int stopX = std::min(int(g.shape(0)), x+ r + 1);
        const int stopY = std::min(int(g.shape(1)), y+ r + 1);
        Coord2d start = {size_t(startX),size_t(startY)};
        Coord2d shape = {size_t(stopX-startX),size_t(stopY-startY)};
        return std::pair<Coord2d, Coord2d>(start,shape);
    }


    const LIFTED_MC_MODEL & model_;
    Settings settings_;

    // rand gen
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<> nodeDist_;
    std::uniform_int_distribution<> edgeDist_;
    std::vector<size_t> globalToLocal_;
    std::vector<size_t> localToGlobal_;
};

template<class LIFTED_MC_MODEL>
class RescaledProposalGenerator{
public:
    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef RescaledProposalGenerator<OTHER_LIFTED_MC_MODEL> type;
    };
    typedef GridGraph<2> GridGraph2D;
    typedef typename GridGraph2D::VertexCoordinate Coord2d;
    typedef std::vector<uint8_t>  LiftedEdgeLabels;
    struct  Settings{
        float reducingFactorMean;
        float reducingFactorSigma;
        int seed;
        Settings()
        :   reducingFactorMean(3.0),
            reducingFactorSigma(2.0),
            seed(-1)
        {

        }
        template<class OTHER>
        Settings(const OTHER & other){
            this->reducingFactorMean = other.reducingFactorMean;
            this->reducingFactorSigma = other.reducingFactorSigma;
            this->seed = other.seed;
        }
    };

    RescaledProposalGenerator(const LIFTED_MC_MODEL & model, const Settings & settings)
    :   model_(model),
        settings_(settings),
        rd_(),
        gen_(settings.seed == -1 ? std::mt19937(rd_()): std::mt19937(size_t(settings.seed)) ),
        facDist_(settings.reducingFactorMean,settings.reducingFactorSigma)
    {

    }


    void generate(const LiftedEdgeLabels & currentBest, LiftedEdgeLabels & proposal){
        this->rescale(model_.originalGraph(), proposal);
    }

    void rescale(const GridGraph2D & graph,LiftedEdgeLabels & proposal){

        

        const auto & liftedGraph = model_.liftedGraph();
        const auto & originalGraph = model_.originalGraph();
        const auto & edgeCosts = model_.edgeCosts();
        auto shapeX = originalGraph.shape(0);
        auto shapeY = originalGraph.shape(1);
        auto fShapeX = float(shapeX);
        auto fShapeY = float(shapeY);
        auto f = facDist_(gen_);
        f = std::max(1.5,f);
        auto iShapeX = int(fShapeX/f + 0.5);
        auto iShapeY = int(fShapeY/f + 0.5);
        
        vigra::TinyVector<int, 2> shape(shapeX,shapeY);
        vigra::TinyVector<int, 2> ishape(iShapeX,iShapeY);
        vigra::MultiArray<2, float> hval(shape,0.0);
        vigra::MultiArray<2, float> vval(shape,0.0);
        vigra::MultiArray<2, float> ihval(ishape,0.0);
        vigra::MultiArray<2, float> ivval(ishape,0.0);
        auto node = [&](const int x,const int y){
            return x+y*shapeX;
        };
        for(auto y=0; y<shapeY; ++y)
        for(auto x=0; x<shapeX; ++x){
            if(x+1<shapeX){
                const auto e = liftedGraph.findEdge(node(x,y),node(x+1,y)).second;
                hval(x,y) += 0.5f *model_.edgeCosts()[e];
                hval(x+1,y) += 0.5f *model_.edgeCosts()[e];
            }
            if(y+1<shapeY){
                const auto e = liftedGraph.findEdge(node(x,y),node(x,y+1)).second;
                vval(x,y) += 0.5f * model_.edgeCosts()[e];
                vval(x,y+1) += 0.5f * model_.edgeCosts()[e];
            }
        }

        vigra::MultiArray<2, float> hvals(hval.shape());
        vigra::MultiArray<2, float> vvals(vval.shape());

        vigra::gaussianSmoothing(hval, hvals, 1.0);
        vigra::gaussianSmoothing(vval, vvals, 1.0);

        vigra::resizeImageSplineInterpolation(hvals, ihval, vigra::BSpline<2, float>());
        vigra::resizeImageSplineInterpolation(vvals, ivval, vigra::BSpline<2, float>());


        GridGraph2D::VertexCoordinate graph_shape;
        graph_shape[0] = iShapeX;
        graph_shape[1] = iShapeY;
        GridGraph2D iGridGraph(graph_shape);
        LiftedMcModel<GridGraph2D,float> iModel(iGridGraph);


        // add long range costs
        auto inode = [&](const vigra::TinyVector<int,2> & coord_){
            return coord_[0]+coord_[1]*iShapeX;
        };

        auto fac = shape/ishape;

        for(auto iy=0; iy<iShapeY; ++iy)
        for(auto ix=0; ix<iShapeX; ++ix){
            if(ix+1<iShapeX){
                const auto val = 0.5f*(ihval(ix,iy) + ihval(ix+1,iy));
                iModel.setCost(
                    inode(vigra::TinyVector<int,2>(ix,iy) ),
                    inode(vigra::TinyVector<int,2>(ix+1,iy) ),
                    val
                );
            }
            if(iy+1<iShapeY){
                const auto val = 0.5f*(ivval(ix,iy) + ivval(ix,iy+1));
                iModel.setCost(
                    inode(vigra::TinyVector<int,2>(ix,iy) ),
                    inode(vigra::TinyVector<int,2>(ix,iy+1) ),
                    val
                );
            }
        }
        for(auto edge =0; edge < liftedGraph.numberOfEdges(); ++edge){
            auto v0 = liftedGraph.vertexOfEdge(edge, 0);
            auto v1 = liftedGraph.vertexOfEdge(edge, 1);
            if(!originalGraph.findEdge(v0,v1).first){

                auto v0y = v0/shapeX;
                auto v0x = v0 - v0y*shapeX;
                auto v1y = v1/shapeX;
                auto v1x = v1 - v1y*shapeX;

                auto fic0 = vigra::TinyVector<float,2>(v0x, v0y)/fac + 0.5f;
                auto fic1 = vigra::TinyVector<float,2>(v1x, v1y)/fac + 0.5f;
                auto ic0 = vigra::TinyVector<int,2>(fic0);
                auto ic1 = vigra::TinyVector<int,2>(fic1);
                if(ic0 != ic1 && ic0[0]>=0 && ic0[0]<iShapeX && ic0[1]>=0 && ic0[1]<iShapeY && 
                                 ic1[0]>=0 && ic1[0]<iShapeX && ic1[1]>=0 && ic1[1]<iShapeY){
                    //std::cout<<"ic0 "<<ic0<<" ic1 "<<ic1<<"\n";
                    auto iu = inode(ic0);
                    auto iv = inode(ic1);
                    iModel.setCost(iu,iv,edgeCosts[edge]);
                }
            }
        }

        // solve model on this scale
        std::vector<uint8_t> subgraphInput(iModel.liftedGraph().numberOfEdges(),1);
        std::vector<uint8_t> subgraphRes(iModel.liftedGraph().numberOfEdges(),1);
        subgraphRes.resize(iModel.liftedGraph().numberOfEdges());
        KernighanLinSettings s;
        s.verbose = false;
        kernighanLin(iModel.originalGraph(),iModel.liftedGraph(),
                     iModel.edgeCosts(),subgraphInput, subgraphRes,s);



        struct SubgraphWithCut { // a subgraph with cut mask
            SubgraphWithCut(const std::vector<uint8_t> & labels, 
                            std::vector<std::size_t> const& edge_in_lifted_graph)
                : labels_(labels), edge_in_lifted_graph_(edge_in_lifted_graph)
                {}
            bool vertex(const std::size_t v) const
                { return true; }
            bool edge(const std::size_t e) const
                { return labels_[edge_in_lifted_graph_[e]] == 0; }

            std::vector<std::size_t> const& edge_in_lifted_graph_;
            const std::vector<uint8_t> & labels_;
        };


        std::vector<size_t> edgeInLiftedGraph(iModel.originalGraph().numberOfEdges());
        for (std::size_t i = 0; i < iModel.originalGraph().numberOfEdges(); ++i){
            auto v0 = iModel.originalGraph().vertexOfEdge(i, 0);
            auto v1 = iModel.originalGraph().vertexOfEdge(i, 1);
            edgeInLiftedGraph[i] = iModel.liftedGraph().findEdge(v0, v1).second;
        } 

        vigra::MultiArray<2, uint64_t> inodeLabels(ishape);
        vigra::MultiArray<2, uint64_t> nodeLabels(shape);

        ComponentsBySearch<GridGraph2D > components;
        components.build(iModel.originalGraph(), SubgraphWithCut(subgraphRes, edgeInLiftedGraph));
        for(std::size_t n=0; n<iModel.originalGraph().numberOfVertices(); ++n){
            auto iy = n/iShapeX;
            auto ix = n - iy*iShapeX;
            auto l = components.labels_[n];
            inodeLabels(ix,iy) = l;
        }

        vigra::resizeImageNoInterpolation(inodeLabels, nodeLabels);

        for(auto edge =0; edge < liftedGraph.numberOfEdges(); ++edge){
            auto v0 = liftedGraph.vertexOfEdge(edge, 0);
            auto v1 = liftedGraph.vertexOfEdge(edge, 1);
            auto v0y = v0/shapeX;
            auto v0x = v0 - v0y*shapeX;
            auto v1y = v1/shapeX;
            auto v1x = v1 - v1y*shapeX;
            proposal[edge] = nodeLabels(v0x,v0y) != nodeLabels(v1x,v1y)  ? 1 : 0 ;
        }

        
    }
    template<class GRAPH>
    void rescale(const GRAPH & graph,LiftedEdgeLabels & proposal){

    }

private:

   

    const LIFTED_MC_MODEL & model_;
    Settings settings_;

    // rand gen
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<> facDist_;
};


template<class LIFTED_MC_MODEL>
class RandomizedProposalGenerator{
public:
    typedef typename LIFTED_MC_MODEL::OriginalGraph OriginalGraph;
    typedef typename LIFTED_MC_MODEL::LiftedGraph LiftedGraph;
    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef RandomizedProposalGenerator<OTHER_LIFTED_MC_MODEL> type;
    };
    typedef GridGraph<2> GridGraph2D;
    typedef typename GridGraph2D::VertexCoordinate Coord2d;
    typedef std::vector<uint8_t>  LiftedEdgeLabels;
    struct  Settings{
        double sigma ;
        double nodeLimit;
        int seed ;
        bool useGA;

        Settings()
        :   sigma(15.0),
            nodeLimit(0.05),
            seed(-1),
            useGA(true)
        {

        }

        template<class OTHER>
        Settings(const OTHER & other){
            this->sigma = other.sigma;
            this->nodeLimit = other.nodeLimit;
            this->seed = other.seed;
            this->useGA = other.useGA;
        }
    };

    RandomizedProposalGenerator(const LIFTED_MC_MODEL & model, const Settings & settings)
    :   model_(model),
        settings_(settings),
        nEdgeCosts_(model.liftedGraph().numberOfEdges()),
        rd_(),
        gen_(settings.seed == -1 ? std::mt19937(rd_()): std::mt19937(size_t(settings.seed)) ),
        nDist_(0,settings.sigma)
    {

    }


    void generate(const LiftedEdgeLabels & currentBest, LiftedEdgeLabels & proposal){

        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts = model_.edgeCosts();

        for(auto e=0; e<model_.liftedGraph().numberOfEdges(); ++e){
            nEdgeCosts_[e] =edgeCosts[e]+nDist_(gen_);
        }

        typedef  LiftedMcModelView< OriginalGraph,LiftedGraph, std::vector<float>  >  NoisyModel;
        NoisyModel noisyModel(originalGraph, liftedGraph, edgeCosts);

        if(settings_.useGA){
            int stopCond = -1;
            if(settings_.nodeLimit<1.0)
                stopCond = double(originalGraph.numberOfVertices())*settings_.nodeLimit + 0.5;
            else
                stopCond = settings_.nodeLimit + 0.5;
            typedef GreedyAdditiveEdgeContraction<NoisyModel>  Solver;
            typedef typename Solver::Settings SolverSettings;
            SolverSettings s;
            s.nodeStopCond = settings_.nodeLimit;
            Solver solver(noisyModel, s);
            std::vector<uint8_t> ones(currentBest.size(), 1);
            solver.run(ones, proposal);
            //greedyAdditiveEdgeContraction(originalGraph,liftedGraph,nEdgeCosts_, proposal, stopCond);
        }
        else{
            KernighanLinSettings s;
            s.verbose = false;
            std::vector<uint8_t> ones(currentBest.size(), 1);
            kernighanLin(originalGraph,liftedGraph,nEdgeCosts_,ones, proposal,s);

        }
    }

private:
    const LIFTED_MC_MODEL & model_;
    Settings settings_;

    // rand gen
    std::random_device rd_;
    std::mt19937 gen_;
    std::normal_distribution<> nDist_;

    std::vector<float> nEdgeCosts_;
};





template<class LIFTED_MC_MODEL, class PROPOSAL_GEN>
class ParallelSolver{
public:
    typedef LIFTED_MC_MODEL LiftedMcModel;
    typedef PROPOSAL_GEN ProposalGen;
    typedef FusionMover<LiftedMcModel> Fm;
    
    class StateBuffer{
    public:
        typedef std::vector<uint8_t> SolVec;
        typedef std::vector<SolVec> SolVecVec;

        StateBuffer(const size_t size, const size_t subSize)
        :   a_(size,SolVec(subSize)),
            b_(size,SolVec(subSize)),
            useA_(true),
            subSize_(subSize){
        }

        
        const SolVec & oldState(const size_t i)const{
            return useA_ ? a_[i] : b_[i];
        }

        SolVec & newState(const size_t i){
            return useA_ ? b_[i] : a_[i];
        }

        void toggle(){
            useA_ = !useA_;
        }

    private:

        SolVecVec a_;
        SolVecVec b_;
        bool useA_;
        size_t subSize_;
    };

    template<class OTHER_LIFTED_MC_MODEL>
    struct Rebind{
        typedef typename PROPOSAL_GEN:: template Rebind<OTHER_LIFTED_MC_MODEL>::type RebindGen;
        typedef ParallelSolver<
            OTHER_LIFTED_MC_MODEL, RebindGen
        > type;
    };


    typedef typename ProposalGen::Settings ProposalGenSettings;


    struct Settings{

        Settings()
        :   
            maxNumberOfIterations(4),
            nParallelProposals ( 100 ),
            reduceIterations (1),
            seed(-1),
            verbose(1),
            proposalsGenSettings(),
            externalProposals(),
            decompose(true),
            nThreads(-1),
            fmKlTimeLimit(std::numeric_limits<double>::infinity()),
            stopIfNotImproved(2)
        {

        }

        template<class OTHER>
        Settings(OTHER & other)
        :   
            maxNumberOfIterations(other.maxNumberOfIterations),

            nParallelProposals(other.nParallelProposals),
            reduceIterations(other.reduceIterations),
            seed(other.seed),
            verbose(other.verbose),
            proposalsGenSettings(other.proposalsGenSettings),
            externalProposals(other.externalProposals),
            decompose(other.decompose),
            nThreads(other.nThreads),
            fmKlTimeLimit(other.fmKlTimeLimit),
            stopIfNotImproved(other.stopIfNotImproved)
        {

        }



        std::size_t maxNumberOfIterations;
        std::size_t nParallelProposals;
        std::size_t reduceIterations;
        int seed ;
        std::size_t verbose;
        ProposalGenSettings proposalsGenSettings;
        std::vector< std::vector<uint8_t> > externalProposals;
        bool decompose;
        int nThreads;
        double fmKlTimeLimit;
        std::size_t stopIfNotImproved;

    };

    ParallelSolver(const LiftedMcModel & model, const Settings & settings = Settings())
    :   model_(model),
        settings_(settings),
        bestEdgeLabels_(model.liftedGraph().numberOfEdges(),0),
        stateBuffer_(settings.nParallelProposals + settings.externalProposals.size(),model_.liftedGraph().numberOfEdges()),
        proposalGens_(),
        fms_(),
        nProposals_(0)
    {

        // allocate proposal generators
        auto pOpt = ParallelOptions();
        const auto nThreads = pOpt.numThreads(settings_.nThreads).getActualNumThreads();
        settings_.nThreads = nThreads;

        proposalGens_.resize(nThreads,nullptr);
        fms_.resize(nThreads,nullptr);
        auto pgs = 0;
        for(auto & pg : proposalGens_){
            auto s = settings_.proposalsGenSettings;
            if(settings_.seed == -1)
                s.seed = -1;
            else
                s.seed  = settings_.seed + pgs;
            pg = new ProposalGen(model_, s);
            ++pgs;
        }

        typedef typename Fm::Settings FMSettings;
        FMSettings fmSettings;
        fmSettings.decompose = settings_.decompose;
        fmSettings.timeLimit = settings_.fmKlTimeLimit;
        // allocate fusion mover
        for(auto & fm : fms_){
            fm = new Fm(model_, fmSettings);
        }
        
    }

    ~ParallelSolver(){
        for(auto & pg : proposalGens_)
            delete pg;
        for(auto fm : fms_)
            delete fm;
    }

        


    template<class LABELS_IN, class LABELS_OUT>
    void run(
        const LABELS_IN & inputEdgeLabels,
        LABELS_OUT & outputLabels
    ){
        // shortcuts
        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts = model_.edgeCosts();

        bool hasStaringPoint = std::any_of(inputEdgeLabels.begin(), inputEdgeLabels.end(), 
            [](uint8_t val){ return val==1; }
        );
        std::copy(inputEdgeLabels.begin(), inputEdgeLabels.end(),  bestEdgeLabels_.begin());


        // setup threadpool
        auto pOpt = ParallelOptions();
        const auto nThreads = pOpt.numThreads(settings_.nThreads).getActualNumThreads();
        ThreadPool threadpool(nThreads);



        bestEdgeLabelsEnergy_ = getEnergy(bestEdgeLabels_);
        auto nWithoutImprovement = 0;

        
        for(auto outerIter=0; outerIter<settings_.maxNumberOfIterations; ++outerIter){

            const auto ePre = bestEdgeLabelsEnergy_;

            this->generateProposals(threadpool);
            this->reduceProposals(threadpool);
            this->fuseTogether(threadpool,outerIter, hasStaringPoint);

            const auto ePost = bestEdgeLabelsEnergy_;

            if(ePost>=ePre)
                ++nWithoutImprovement;
            else
                nWithoutImprovement = 0;
            if(nWithoutImprovement >= settings_.stopIfNotImproved)
                break;
           
        }
        std::copy(bestEdgeLabels_.begin(), bestEdgeLabels_.end(), outputLabels.begin());

    }
    const LiftedMcModel & getModel()const{
        return model_;
    }
private:







    template<class EDGE_LABELS>
    double getEnergy(const EDGE_LABELS & edgeLabels_) {
        auto totalCost = 0.0;
        for(std::size_t edge = 0; edge < model_.liftedGraph().numberOfEdges(); ++edge){
            if(edgeLabels_[edge]){
                totalCost += model_.edgeCosts()[edge];
            }
        }
        return totalCost;
    };


    void generateProposals(ThreadPool & threadpool){

        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts = model_.edgeCosts();

        ////////////////////////////////////////////////
        // Generate proposals in parallel
        ////////////////////////////////////////////////
        if(settings_.verbose >=2)
            std::cout<<"generate proposals..\n";
        {
            auto nParallelProposals = settings_.nParallelProposals;
            auto nEdges = model_.liftedGraph().numberOfEdges();
            std::function<void(int)> cb;
            ProgressCallback progressCallback(nParallelProposals, cb);
            progressCallback.setVerboseCallback(settings_.verbose);
            parallel_foreach(threadpool, nParallelProposals,[&](const int threadId, const int p){


                // generate the proposal
                auto & proposal = stateBuffer_.newState(p);

                std::fill(proposal.begin(), proposal.end(), 0);
                proposalGens_[threadId]->generate(bestEdgeLabels_, proposal);
                

                // report progress
                progressCallback.increment(1);

            });
            //if(settings_.verbose >=1);
            //    std::cout<<"...done\n";
        }
        for(auto i=0; i<settings_.externalProposals.size(); ++i){
            auto & proposal = stateBuffer_.newState(i+settings_.nParallelProposals);
            const auto & extProb = settings_.externalProposals[i];
            std::copy(extProb.begin(), extProb.end(), proposal.begin());
        }
        nProposals_ = settings_.nParallelProposals + settings_.externalProposals.size();
        stateBuffer_.toggle();
    }

    void reduceProposals(ThreadPool & threadpool){
        if(settings_.reduceIterations > 1){
            //std::cout<<"pEnd_"<<pEnd_<<" "<<edgeLabelsBuffer_.size()<<"\n";

            int newNProposals = 0;
            const auto nEdge = model_.liftedGraph().numberOfEdges();
            const auto combineN = settings_.reduceIterations;
            for(auto i=0; i<nProposals_; i+=combineN){
                auto stop = std::min(i+int(combineN),nProposals_);
                auto & out = stateBuffer_.newState(newNProposals);
                std::fill(out.begin(), out.end(), 0);
                for(auto j=i; j<stop; ++j){
                    for(auto e=0; e<nEdge; ++e){
                        out[e] = std::max(out[e],stateBuffer_.oldState(j)[e]);
                    }
                }
                ++newNProposals;
            }
            nProposals_ = newNProposals;
            stateBuffer_.toggle();
            std::cout<<"reduce done\n";
        }
    }

    void fuseTogether(ThreadPool & threadpool, const int outerIter, const bool hasStaringPoint){

        const auto & originalGraph = model_.originalGraph();
        const auto & liftedGraph = model_.liftedGraph();
        const auto & edgeCosts = model_.edgeCosts();
        std::mutex toFuseMutex;
        if(settings_.verbose>=2)
            std::cout<<"fuse with best \n";

        if(outerIter>=1 || hasStaringPoint)
        {
            //if(settings_.verbose >=1)
            //    std::cout<<"fuse proposals..\n";
            //ProgressCallback progressCallback(toFuse.size());
            //progressCallback.setVerboseCallback(settings_.verbose);
            int newNProposals = 0;
            auto nJobs = nProposals_;

            parallel_foreach(threadpool, nJobs,[&](const int threadId, const int p){

                auto fm = fms_[threadId];
                auto & proposal = stateBuffer_.oldState(p);
                auto proposalsEnergy = getEnergy(proposal);

                // copy best in a locked fashion
                std::vector<uint8_t> bestCopy;
                double bestECopy;
                {
                    std::unique_lock<std::mutex> lock(toFuseMutex);
                    bestCopy = bestEdgeLabels_;
                    bestECopy = getEnergy(bestCopy);
                }

                std::vector<uint8_t> fuseResults(liftedGraph.numberOfEdges());
                auto fuseEnergy = fm->fuse(bestCopy, proposal,bestECopy,proposalsEnergy, fuseResults);
                if(fuseEnergy < bestECopy)
                {
                    {
                        std::unique_lock<std::mutex> lock(toFuseMutex);
                        stateBuffer_.newState(newNProposals) = fuseResults;
                        bestEdgeLabels_ = fuseResults;
                        bestEdgeLabelsEnergy_ = fuseEnergy;
                        ++newNProposals;
                    }
                }
                // report progress
                //progressCallback.increment(1);
            });
            nProposals_ =newNProposals;
            stateBuffer_.toggle();
        }



        auto nToFuse = nProposals_;
        if(settings_.verbose>=2)
            std::cout<<"hierarchical Fusion "<<nToFuse<<"\n";
        if(nToFuse > 0){

            

            auto level = 0;
            while(true){
                //std::set<double, own_double_less> valSet;
                //std::cout<<"LEVEL "<<level<<"\n";
                //std::cout<<"  nProposals_ "<<nProposals_<<"\n";
                //std::cout<<"  pEnd_ "<<pEnd_<<"\n";
                //std::cout<<"  d "<<pEnd_ - pStart_<<"\n";
              
                int newNProposals = 0;             
                nToFuse = nProposals_;
                if(nToFuse == 0 ){
                    GRAPH_CHECK(false,"");
                }
                if(nToFuse == 1){
                    stateBuffer_.newState(newNProposals) = stateBuffer_.oldState(0);
                    ++newNProposals;
                    nProposals_ = 1;
                    stateBuffer_.toggle();
                    break;
                }
                if(nToFuse%2 == 1){
                    //std::cout<<"INSERTED ONE \n";
                    //--pEnd_;
                    // toFuse2.push_back(toFuse.back());
                    // toFuse.pop_back();
                    stateBuffer_.newState(newNProposals) = stateBuffer_.oldState(nProposals_-1);
                    //valSet.insert(getEnergy(edgeLabelsBuffer_[pEnd_-1]));
                    ++newNProposals;
                }
                auto nJobs = nToFuse/2;
                //if(settings_.verbose >=1)
                    ////std::cout<<"hierarchical..\n";
                //
                
               
                //ProgressCallback progressCallback(nJobs);
                //progressCallback.setVerboseCallback(settings_.verbose);
                parallel_foreach(threadpool, nJobs  ,[&](const int threadId, const int i){
                    GRAPH_CHECK_OP(i*2,<,nProposals_,"");
                    GRAPH_CHECK_OP(i*2+1,<,nProposals_,"");
                    auto fm = fms_[threadId];
                    auto & pa = stateBuffer_.oldState(i*2);
                    auto & pb = stateBuffer_.oldState(i*2+1);
                    std::vector<uint8_t> fuseResults(liftedGraph.numberOfEdges());
                    const auto fuseEnergy = fm->fuse(pa, pb, fuseResults);
                    {
                        //if(valSet.find(fuseEnergy)==valSet.end()){
                            //valSet.insert(fuseEnergy);
                            std::unique_lock<std::mutex> lock(toFuseMutex);
                            stateBuffer_.newState(newNProposals) = fuseResults;
                            //std::cout<<"   FR "<<getEnergy(fuseResults)<<"\n";
                            ++newNProposals;
                        //}
                        //else{
                            //std::cout<<"saved computation\n";
                        //}
                    }
                    //progressCallback.increment(1);
                });
                nProposals_ = newNProposals;
                stateBuffer_.toggle();

                ++level;
            }
            GRAPH_CHECK_OP(nProposals_,==,1,"");    
            bestEdgeLabels_ = stateBuffer_.oldState(0);
            bestEdgeLabelsEnergy_ = getEnergy(bestEdgeLabels_);
        }


    }


    LiftedMcModel model_;
    Settings settings_;
    std::vector<uint8_t> bestEdgeLabels_;

    StateBuffer stateBuffer_;


    std::vector<ProposalGen*> proposalGens_;
    std::vector<Fm*> fms_;
    int nProposals_;
    double bestEdgeLabelsEnergy_;

};






} // end namespace multicut-lifted
} // end namespace graph
} // end namespace andres

#endif /*ANDRES_GRAPH_MULTICUT_PARALLEL_LIFTED_MC_HXX*/
