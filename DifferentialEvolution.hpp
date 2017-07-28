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
//  DifferentialEvolution.hpp
//

#ifndef DifferentialEvolution_hpp
#define DifferentialEvolution_hpp

#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>


struct data {
    //float *arr;
    float v;
    int dim;
};

class DifferentialEvolution {
private:
    float *d_target1;
    float *d_target2;
    float *d_cost;
    float *d_mutant;
    float *d_trial;
    
    
    float *d_min;
    float *d_max;
    float *h_cost;
    
    void *d_randStates;
    
    int popSize;
    int dim;
    
    int CR;
    int numGenerations;
    float F;
    
public:
    
    DifferentialEvolution(int PopulationSize, int NumGenerations, int Dimensions,
                float crossoverConstant, float mutantConstant,
                float *minBounds, float *maxBounds);
    
    std::vector<float> fmin(void *args);
    
};

#endif /* DifferentialEvolution_hpp */
