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

#include <stdio.h>


#include "DifferentialEvolution.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>



int main(void)
{
    
    float minBounds[2] = {-5,-24.3};
    float maxBounds[2] = {75,101.4};
    
    struct data x;
    struct data d_x;
    cudaMalloc(&x.arr, sizeof(float) * 10);
    unsigned long size = sizeof(struct data);
    cudaMalloc((void *)&d_x, size);
    x.v = 0.5;
    x.dim = 2;
    
    DifferentialEvolution minimizer(192,500, 2, 0.9, 0.5, minBounds, maxBounds, d_func);
    
    cudaMemcpy(&d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice);
    std::vector<float> result = minimizer.fmin(&d_x);
    std::cout << "Result = " << result[0] << ", " << result[1] << std::endl;
    std::cout << "Finished main function." << std::endl;
    return 1;
}
