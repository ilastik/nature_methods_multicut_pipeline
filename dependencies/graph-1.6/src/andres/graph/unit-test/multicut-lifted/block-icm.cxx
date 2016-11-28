#include <iostream>
#include <stdexcept>
#include <random>

#include "andres/runtime-check.hxx"
#include "andres/graph/graph.hxx"
#include "andres/graph/grid-graph.hxx"
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/multicut-lifted/block-icm-lmc2.hxx"
#include "andres/graph/multicut-lifted/lifted_mc_model.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"



using namespace andres::graph;


/*
    so far this does not really test something

    STILL BUGGY
*/
void testMulticutLifted()
{
    typedef GridGraph<2>  G;
    typedef multicut_lifted::LiftedMcModel<GridGraph<2>, float>  Model;

    auto sx = 100;
    auto sy = 100;
    auto r = 2;

    auto node = [&](int x, int y){
        return x + y*sx;
    };

    GridGraph<2> graph({size_t(sx), size_t(sy)});
    Model model(graph);



    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-5, 5);

    for(int ii=0; ii<20; ++ii){

        if(r>0){
            for(auto y=0; y<sy; ++y)
            for(auto x=0; x<sx; ++x){
                auto u = node(x,y);

                for(auto yy=y-r; yy<y+r+1; ++yy)
                for(auto xx=x-r; xx<x+r+1; ++xx){
                    if(xx>=0 && yy>=0 && xx<sx && yy<sy){
                        auto v = node(xx,yy);
                        if(u<v){
                            model.setCost(u,v,distribution(generator));
                        }
                    }
                }
            }   
        }

        for(auto y=0; y<sy; ++y)
        for(auto x=0; x<sx; ++x){
            auto u = node(x,y);

            if(x+1 < sx){
                auto v = node(x+1,y);
                model.setCost(u,v,distribution(generator));
            }
            if(y+1 < sy){
                auto v = node(x,y+1);
                model.setCost(u,v,distribution(generator));
            }
        }   
        // lambda to compute energy 
        auto getEnergy = [&] (const std::vector<uint8_t> & edgeLabels_) {
            auto totalCost = 0.0;
            for(std::size_t edge = 0; edge < model.liftedGraph().numberOfEdges(); ++edge){
                if(edgeLabels_[edge]){
                    totalCost += model.edgeCosts()[edge];
                }
            }
            return totalCost;
        };

        // set up the solver
        typedef multicut_lifted::KernighanLin<Model> SubSolver;
        typedef typename SubSolver::Settings SubSolverSettings;
        typedef multicut_lifted::BlockIcmLiftedMc<SubSolver> Solver;
        typedef typename Solver::Settings SolverSettings;

        std::vector<uint8_t> in(model.liftedGraph().numberOfEdges(),1);
        std::vector<uint8_t> out(model.liftedGraph().numberOfEdges(),0);
        std::vector<uint8_t> out2(model.liftedGraph().numberOfEdges(),0);
        SubSolverSettings subSolverSettings;
        subSolverSettings.verbose = true;


        SubSolver subSolver(model,subSolverSettings);
        subSolver.run(in, out);
        std::cout<<"E KL "<<getEnergy(out)<<"\n";
        
        subSolverSettings.verbose = false;
        SolverSettings solverSettings;
        solverSettings.solverSettings = subSolverSettings;
        solverSettings.numberOfIterations = 200;
        solverSettings.subgraphRadius = 30;
        Solver solver(model, solverSettings);
        solver.run(out,out2);
        std::cout<<"E ICM "<<getEnergy(out2)<<"\n";
    }
}

int main()
{
    testMulticutLifted();

    return 0;
}
