# CUDADifferentialEvolution

TODO: write better readme
    : Add random vector if agent is outside search space
    : Consider usage of standard deviation to stop execution before running all generations.

This repo holds the GPU kernel functions required to run differential evolution.
The software in this files is based on the paper:
Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continous Spaces,
Rainer Storn, Kenneth Price (1996)

But is extended upon for use with GPU's for faster computation times.
This has been done previously in the paper:
Differential evolution algorithm on the GPU with C-CUDA
Lucas de P. Veronese, Renato A. Krohling (2010)
However this implementation is only vaguly based on their implementation.
Translation: I saw that the paper existed, and figured that they probably
implemented the code in a similar way to how I was going to implement it.
Brief read-through seemed to be the same way.

The paralization in this software is done by using multiple cuda threads for each
agent in the algorithm. If using smaller population sizes, (4 - 31) this will probably
not give significant if any performance gains. However large population sizes are more
likly to give performance gains.

HOW TO USE:
To implement a new cost function write the cost function in DifferentialEvolutionGPU.cu with the header
__device float fooCost(const float *vec, const void *args)
@param vec - sample parameters for the cost function to give a score on.
@param args - any set of arguements that can be passed at the minimization stage
NOTE: args any memory given to the function must already be in device memory.

Go to the header and add a specifier for your cost functiona and change the COST_SELECTOR
to that specifier. (please increment from previous number)

Once you have a cost function find the costFunc function, and add into
preprocessor directives switch statement

...
#elif COST_SELECTOR == YOUR_COST_FUNCTION_SPECIFIER
     return yourCostFunctionName(vec, args);
...
