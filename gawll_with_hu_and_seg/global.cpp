#include "defs.h"

// Global variables and parameters
int chrom_size;										// size of the chromosome (dimension of the problem) 
int gen;											// generation
int LS_flag=0;										// with (1) or without (0) local search 
int mutLL_flag;										// with (1) or without (0) mutation LL
int n_runs_per_instance = 2;						// number of runs 
int popsize=10;									// size of the population 
int save_datagen_flag=0;							// flag for saving data for generation in the first run
int tau_reset=10;									// number of generations for reseting part of the population and applying local search
int tournament_size=3;								// size of the pool for tournament selection 	
int n_edges_eVIG;									// number of edges in the eVIG	
int total_edges_VIG;								// total of edes in the VIG
long int max_gen;									// maximum number of generations (when stop criterion is time, it is used for saving some statistics data)
double p_cross=0.4;									// crossover rate											
double p_mut;										// mutation rate
double resetpop_rate=0.99;							// controls the percentage of new individuals reset
population popold , popnew;							// population (new and old)
// Vectors
int *vsort_aux;										// auxiliar sorted vector of integers (used in different methods)
int *file_gen;										// data to be stored: number of generations								
double *file_best_fitness, *time_run;				// data to be stored: best fitness, runtime
double *file_best_fitness_gen;						// data to be stored: best fitness over the generations for run 0
double *file_n_edges_eVIG;							// data to be stored: number of edges of the eVIG
double *file_n_edges_eVIG_gen;						// data to be stored: number of edges of the eVIG over the generations for run 0
// Matrices
int **File_best_ind;								// data to be stored: best individual
// Feature Selection problem
double **X_trainset, **X_testset;					// train and test sets: inputs
double *d_trainset_regression, *d_testset_regression;// train and test sets: desired outputs  - Regression problems
int *d_trainset, *d_testset;						// train and test sets: desired outputs  - Classification problems
int n_examples_train, n_examples_test;				// number of examples in the training and test sets
int n_classes;										// number of classes
int classifier_type;								// classifier 1: KNN with K=3; 2: KNN with K=5 ; 3: not defined yet
int prob_type;										// problem type: 1 - classification; 2 - regression

