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
//  testMain.cpp
//
// This is a test code to show an example usage of Differential Evolution

#include <stdio.h>

#include "DifferentialEvolution.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>



int main(void)
{
    // create the min and max bounds for the search space.
    float minBounds[2] = {-50, -50};
    float maxBounds[2] = {100, 200};
    
    // a random array or data that gets passed to the cost function.
    float arr[3] = {2.5, 2.6, 2.7};
    
    // data that is created in host, then copied to a device version for use with the cost function.
    struct data x;
    struct data *d_x;
    gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * 3));
    unsigned long size = sizeof(struct data);
    gpuErrorCheck(cudaMalloc((void **)&d_x, size));
    x.v = 3;
    x.dim = 2;
    gpuErrorCheck(cudaMemcpy(x.arr, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
    
    // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
    DifferentialEvolution minimizer(192,50, 2, 0.9, 0.5, minBounds, maxBounds);
    
    gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
    
    // get the result from the minimizer
    std::vector<float> result = minimizer.fmin(d_x);
    std::cout << "Result = " << result[0] << ", " << result[1] << std::endl;
    std::cout << "Finished main function." << std::endl;
    return 1;
}
