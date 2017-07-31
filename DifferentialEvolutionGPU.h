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

#include "DifferentialEvolution.hpp"
#include <cuda_runtime.h>

#ifndef __DIFFERENTIAL_EVOLUTION_GPU__
#define __DIFFERENTIAL_EVOLUTION_GPU__

#define QUADRATIC_COST 0
#define COST_WITH_ARGS 1


#define COST_SELECTOR COST_WITH_ARGS


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);

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

void *createRandNumGen(int size);

#endif
