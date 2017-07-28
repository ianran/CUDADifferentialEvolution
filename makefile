all:
	nvcc -o output testMain.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu


clean:
	
