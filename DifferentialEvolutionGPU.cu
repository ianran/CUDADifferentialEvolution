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


#include <curand_kernel.h>
#include <cuda_runtime.h>
// for random numbers in a kernel
#include "DifferentialEvolutionGPU.h"

// for FLT_MAX
#include <cfloat>

#include <iostream>

// for clock()
#include <ctime>
#include <cmath>



// -----------------IMPORTANT----------------
// costFunc - this function must implement whatever cost function
// is being minimized.
// Feel free to delete all code in here.
// This is a bit of a hack and not elegant at all. The issue is that
// CUDA doesn't support function passing device code between host
// software. There is a possibilty of using virtual functions, but
// was concerned that the polymorphic function have a lot of overhead
// So instead use this super ugly method for changing the cost function.
//
// @param vec - the vector to be evaulated.
// @param args - a set of user arguments.
__device__ float costFunc(const float *vec, const void *args)
{
    const struct data *a = (struct data *)args;
    float x = vec[0];
    float y = vec[1];
    //return (x*x*x*x)- (2*x*x*x)+25;
    //float z = (2*y)-2;
    //return (x*x*x*x)- (2*x*x*x) + (z*z*z*z) + (y*y*y);
    //return -46.78;
    //return -(cos(x) + cos(y)) + 0.2*(x*x) + 0.2*(y*y);
    return a->arr[2] + (x*x) + (y*y) + a->v;
}













void printCudaVector(float *d_vec, int size)
{
    float *h_vec = new float[size];
    cudaMemcpy(h_vec, d_vec, sizeof(float) * size, cudaMemcpyDeviceToHost);
    
    std::cout << "{";
    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << "}" << std::endl;
    
    delete[] h_vec;
}

__global__ void generateRandomVectorAndInit(float *d_x, float *d_min, float *d_max,
            float *d_cost, CostFunc_t costFuncPassed, void *costArgs, curandState_t *randStates,
            int popSize, int dim, unsigned long seed)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return;
    
    float test[1] = {4};
    
    curandState_t *state = &randStates[idx];
    curand_init(seed, idx,0,state);
    for (int i = 0; i < dim; i++) {
        d_x[(idx*dim) + i] = (curand_uniform(state) * (d_max[i] - d_min[i])) + d_min[i];
    }
    d_cost[idx] = costFunc(&d_x[idx*dim], costArgs);
    //d_cost[idx] = costFunc(test, costArgs);
}

__global__ void evolutionKernel(CostFunc_t costFuncPassed,
                                float *d_target,
                                float *d_trial,
                                float *d_cost,
                                float *d_target2,
                                curandState_t *randStates,
                                int dim,
                                int popSize,
                                int maxGenerations,
                                int CR, // Must be given as value between [0,999]
                                float F,
                                void *costArgs)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return; // stop executing this block if
                                // all populations have been used
    curandState_t *state = &randStates[idx];
    
    // TODO: Better way of generating unique random numbers?
    int a;
    int b;
    int c;
    int j;
    //////////////////// Random index mutation generation //////////////////
    // select a different random number then index
    do { a = curand(state) % popSize; } while (a == idx);
    do { b = curand(state) % popSize; } while (b == idx || b == a);
    do { c = curand(state) % popSize; } while (c == idx || c == a || c == b);
    j = curand(state) % dim;
    
    ///////////////////// MUTATION ////////////////
    for (int k = 1; k <= dim; k++) {
        if ((curand(state) % 1000) < CR || k==dim) {
            // trial vector param comes from vector plus weighted differential
            d_trial[(idx*dim)+j] = d_target[(a*dim)+j] + (F * (d_target[(b*dim)+j] - d_target[(c*dim)+j]));
        } else {
            d_trial[(idx*dim)+j] = d_target[(idx*dim)+j];
        } // end if else for creating trial vector
        j = (j+1) % dim;
    } // end for loop through parameters
    
    float score = costFunc(&d_trial[idx*dim], costArgs);
    if (score < d_cost[idx]) {
        // copy trial into new vector
        for (j = 0; j < dim; j++) { d_target2[(idx*dim) + j] = d_trial[(idx*dim) + j]; }
        d_cost[idx] = score;
    }
} // end differentialEvolution function.

void differentialEvolution(CostFunc_t costFunc,
                           float *d_target,
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
                           float *h_output)
{
    int power32 = ceil(popSize / 32.0) * 32;
    //std::cout << "power32 = " << power32 << std::endl;
    
    // generate random vector
    generateRandomVectorAndInit<<<1, power32>>>(d_target, d_min, d_max, d_cost,
                    costFunc, costArgs, (curandState_t *)randStates, popSize, dim, clock());
    
    cudaMemcpy(d_target2, d_target, sizeof(float) * dim * popSize, cudaMemcpyDeviceToDevice);
    
    //printCudaVector(d_target, popSize*dim);
    //printCudaVector(d_cost, popSize);
    
    for (int i = 1; i <= maxGenerations; i++) {
        //std::cout << i << ": generation = \n";
        //printCudaVector(d_target, popSize * dim);
        //std::cout << "cost = ";
        //printCudaVector(d_cost, popSize);
        //std::cout << std::endl;
        
        // start kernel for this generation
        evolutionKernel<<<1, power32>>>(costFunc, d_target, d_trial, d_cost, d_target2, (curandState_t *)randStates,
                                        dim, popSize, maxGenerations, CR, F, costArgs);
        
        // swap buffers, places newest data into d_target.
        float *tmp = d_target;
        d_target = d_target2;
        d_target2 = tmp;
    } // end for (generations)
    
    cudaMemcpy(h_cost, d_cost, popSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    //std::cout << "h_cost = {";
    
    // find min of last evolutions
    int bestIdx = -1;
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = h_cost[i];
        //std::cout << curCost << ", ";
        if (curCost <= bestCost) {
            bestCost = curCost;
            bestIdx = i;
        }
    }
    //std::cout << "}" << std::endl;
    
    // output best minimization.
    cudaMemcpy(h_output, d_target+bestIdx, sizeof(float)*dim, cudaMemcpyDeviceToHost);
}

// allocate the memory needed for random number generators.
void *createRandNumGen(int size)
{
    void *x;
    cudaMalloc(&x, sizeof(curandState_t)*size);
    return x;
}









