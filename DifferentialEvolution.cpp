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

//
//  DifferentialEvolution.cpp
//

#include "DifferentialEvolution.hpp"
#include "DifferentialEvolutionGPU.h"

// Constructor for DifferentialEvolution
//
// @param PopulationSize - the number of agents the DE solver uses.
// @param NumGenerations - the number of generation the differential evolution solver uses.
// @param Dimensions - the number of dimesnions for the solution.
// @param crossoverConstant - the number of mutants allowed pass each generation CR in
//              literature given in the range [0,1]
// @param mutantConstant - the scale on mutant changes (F in literature) given [0,2]
//              default = 0.5
// @param func - the cost function to minimize.
DifferentialEvolution::DifferentialEvolution(int PopulationSize, int NumGenerations,
        int Dimensions, float crossoverConstant, float mutantConstant,
        float *minBounds, float *maxBounds)
{
    popSize = PopulationSize;
    dim = Dimensions;
    numGenerations = NumGenerations;
    CR = crossoverConstant*1000;
    F = mutantConstant;
    
    cudaMalloc(&d_target1, sizeof(float) * popSize * dim);
    cudaMalloc(&d_target2, sizeof(float) * popSize * dim);
    cudaMalloc(&d_mutant, sizeof(float) * popSize * dim);
    cudaMalloc(&d_trial, sizeof(float) * popSize * dim);
    
    cudaMalloc(&d_cost, sizeof(float) * PopulationSize);
    
    cudaMalloc(&d_min, sizeof(float) * dim);
    cudaMalloc(&d_max, sizeof(float) * dim);
    cudaMemcpy(d_min, minBounds, sizeof(float) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, maxBounds, sizeof(float) * dim, cudaMemcpyHostToDevice);
    
    h_cost = new float[popSize * dim];
    d_randStates = createRandNumGen(popSize);
}

std::vector<float> DifferentialEvolution::fmin(void *args)
{
    std::vector<float> result(dim);
    
    differentialEvolution(d_target1, d_trial, d_cost, d_target2, d_min,
            d_max, h_cost, d_randStates, dim, popSize, numGenerations, CR, F, args,
            result.data());
    
    return result;
}
