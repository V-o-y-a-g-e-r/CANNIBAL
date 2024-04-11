/******************************************************************************\
*								 Definitions							 *
\******************************************************************************/
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <list> 
#define CHAR_LEN 1000
#define EPS2 1.0e-12

using namespace std; 

#ifndef DEF_H
#define DEF_H

// Data structures
typedef int allele; 										// type allele
typedef struct {
			allele *chromosome;								
			double fitness;									
} individual;												// data structure individual
typedef struct {			
			individual *ind;
			double sum_fitness;
			double mean_fitness;
			double max_fitness;
			int best_individual;		
} population;												// data structure population


// Global variables and parameters
extern int chrom_size;										// size of the chromosome (dimension of the problem) 
extern int gen;												// generation
extern int LS_flag;											// with (1) or without (0) local search 
extern int mutLL_flag;										// with (1) or without (0) mutation LL		
extern int n_runs_per_instance;								// number of runs 
extern int popsize;											// size of the population 
extern int save_datagen_flag;								// flag for saving data for generation in the first run
extern int tau_reset;										// number of generations for reseting part of the population and applying local search
extern int tournament_size;									// size of the pool for tournament selection 								
extern int n_edges_eVIG;									// number of edges in the eVIG
extern int total_edges_VIG;									// total of edes in the VIG
extern long int max_gen;									// maximum number of generations (when stop criterion is time, it is used for saving some statistics data)
extern double p_cross;										// crossover rate	
extern double p_mut;										// mutation rate
extern double resetpop_rate;								// controls the percentage of new individuals reset
extern population popold , popnew;							// population (new and old)
// Vectors
extern int *vsort_aux;										// auxiliar sorted vector of integers (used in different methods)
extern int *file_gen;										// data to be stored: number of generations								
extern double *file_best_fitness, *time_run;				// data to be stored: best fitness, runtime
extern double *file_best_fitness_gen;						// data to be stored: best fitness over the generations for run 0
extern double *file_n_edges_eVIG;							// data to be stored: number of edges of the eVIG
extern double *file_n_edges_eVIG_gen;						// data to be stored: number of edges of the eVIG over the generations for run 0
// Matrices
extern int **File_best_ind;									// data to be stored: best individual
// Feature Selection problem
extern double **X_trainset, **X_testset;					// train and test sets: inputs
extern double *d_trainset_regression, *d_testset_regression;// train and test sets: desired outputs  - Regression problems
extern int *d_trainset, *d_testset;							// train and test sets: desired outputs  - Classification problems
extern int n_examples_train, n_examples_test;				// number of examples in the training and test sets
extern int n_classes;										// number of classes
extern int classifier_type;									// classifier 1: KNN with K=3; 2: KNN with K=5; 3: not defined yet
extern int prob_type;										// problem type: 1 - classification; 2 - regression
	
// Function declaration
//  aux_functions.cpp
void desaloc_matrixd(double **Matrix , int lines);
void desaloc_matrixi(int **Matrix , int lines);
void rand_perm_size(int *inp, int *out, int size_inp, int size_out);
int isVectorEqual(int *v1, int *v2, int size_v);
int random_int(int L_range, int H_range);
int *aloc_vectori(int lines);
int **aloc_matrixi(int lines , int collums);
double random_dou(void);
double *aloc_vectord(int lines);
double **aloc_matrixd(int lines , int collums);
individual *aloc_vectorind(int lines);
// file_man.cpp
void file_output(char *prob_name, int total_runs);
void read_problem(char *prob_name);
// statistics.cpp
void checkBestInd(population *pop, int j, int n_run);
void statistics(population *pop, int n_run);
// selection.cpp
int selection(population *pop);
// transformation.cpp
void mutation (allele *offspring);
void Point2X(allele *parent1, allele *parent2, allele *offspring1, allele *offspring2);
void UX(allele *parent1, allele *parent2, allele *offspring1, allele *offspring2);
// fitness.cpp
double compFitFS(allele *ind);

#endif
