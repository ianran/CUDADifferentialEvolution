/* Copyright 2017 Ian Rankin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 * to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// DifferentialEvolutionGPU.h
// This file holds the header to GPU kernel functions required to run differential evolution.
// The software in this files is based on the paper:
// Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continous Spaces,
// Rainer Storn, Kenneth Price (1996)
//
// But is extended upon for use with GPU's for faster computation times.
// This has been done previously in the paper:
// Differential evolution algorithm on the GPU with C-CUDA
// Lucas de P. Veronese, Renato A. Krohling (2010)
// However this implementation is only vaguly based on their implementation.
// Translation: I saw that the paper existed, and figured that they probably
// implemented the code in a similar way to how I was going to implement it.
// Brief read-through seemed to be the same way.
//
// The paralization in this software is done by using multiple cuda threads for each
// agent in the algorithm. If using smaller population sizes, (4 - 31) this will probably
// not give significant if any performance gains. However large population sizes are more
// likly to give performance gains.
//
// HOW TO USE:
// To implement a new cost function write the cost function in DifferentialEvolutionGPU.cu with the header
// __device float fooCost(const float *vec, const void *args)
// @param vec - sample parameters for the cost function to give a score on.
// @param args - any set of arguements that can be passed at the minimization stage
// NOTE: args any memory given to the function must already be in device memory.
//
// Go to the header and add a specifier for your cost functiona and change the COST_SELECTOR
// to that specifier. (please increment from previous number)
//
// Once you have a cost function find the costFunc function, and add into
// preprocessor directives switch statement
//
// ...
// #elif COST_SELECTOR == YOUR_COST_FUNCTION_SPECIFIER
//      return yourCostFunctionName(vec, args);
// ...
//

#include "DifferentialEvolution.hpp"
#include <cuda_runtime.h>

#ifndef __DIFFERENTIAL_EVOLUTION_GPU__
#define __DIFFERENTIAL_EVOLUTION_GPU__

#define QUADRATIC_COST 0
#define COST_WITH_ARGS 1
#define MANY_LOCAL_MINMA 2
#define COST_3D 3


#define COST_SELECTOR COST_WITH_ARGS

// A basic macro used to checking cuda errors.
// @param ans - the most recent enumerated cuda error to check.
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);

// This function handles the entire differentialEvolution, and calls the needed kernel functions.
// @param d_target - a device array with the current agents parameters (requires array with size popSize*dim)
// @param d_trial - a device array with size popSize*dim (worthless outside of function)
// @param d_cost - a device array with the costs of the last generation afterwards size: popSize
// @param d_target2 - a device array with size popSize*dim (worthless outside of function)
// @param d_min - a list of the minimum values for the set of parameters (size = dim)
// @param d_max - a list of the maximum values for the set of parameters (size = dim)
// @param h_cost - this function once the function is completed will contain the costs of final generation.
// @param randStates - an array of random number generator states. Array created using createRandNumGen funtion
// @param dim - the number of dimensions the equation being minimized has.
// @param popSize - this the population size for DE, or otherwise the number of agents that DE will use. (see DE paper for more info)
// @param maxGenerations - the max number of generations DE will perform (see DE paper for more info)
// @param CR - Crossover Constant used by DE (see DE paper for more info)
// @param F - the scaling factor used by DE (see DE paper for more info)
// @param costArgs - this a set of any arguments needed to be passed to the cost function. (must be in device memory already)
// @param h_output - the host output vector of function
void differentialEvolution(float *d_target,
                           float *d_trial,
                           float *d_cost,
                           float *d_target2,
                           float *d_min,
                           float *d_max,
                           float *h_cost,
                           void *randStates,
                           int dim,
                           int popSize,
                           int maxGenerations,
                           int CR, // Must be given as value between [0,999]
                           float F,
                           void *costArgs,
                           float *h_output);

// createRandNumGen
// Inits the array of random number generators required by differentialEvolution
// @param size - this should be the same size as popSize
void *createRandNumGen(int size);

#endif
