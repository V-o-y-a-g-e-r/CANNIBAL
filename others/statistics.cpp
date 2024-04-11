/******************************************************************************\
*								 Statistics						 *
\******************************************************************************/
#include "defs.h"


/******************************************************************************\
*				  	Population Statistics 									   *
\******************************************************************************/
void checkBestInd(population *pop, int j, int n_run){
	// check if j-th ind. of pop improved best individual
	if (pop->ind[j].fitness > file_best_fitness[n_run]){		
		file_best_fitness[n_run] = pop->ind[j].fitness;
		for(int gene=0;gene<chrom_size;gene++)
			File_best_ind[n_run][gene]=pop->ind[j].chromosome[gene];
	}
	
}


/******************************************************************************\
*				  	Population Statistics 									   *
\******************************************************************************/
void statistics(population *pop, int n_run){	

	pop->sum_fitness = pop->ind[0].fitness; 			// sum of the fitness in the population
	pop->max_fitness = pop->ind[0].fitness;   			// maximum fitness in the population
	pop->best_individual = 0;							// best individual in the population
	if (pop->ind[0].fitness > file_best_fitness[n_run]){
			file_best_fitness[n_run] = pop->ind[0].fitness;
			for(int gene=0;gene<chrom_size;gene++) 
				File_best_ind[n_run][gene]=pop->ind[0].chromosome[gene];
	}		
	for(int j=1;j<popsize;j++) {
		pop->sum_fitness = pop->sum_fitness + pop->ind[j].fitness;
		if (pop->ind[j].fitness > pop->max_fitness )	{	
			pop->max_fitness = pop->ind[j].fitness; 
			pop->best_individual = j;			
		}
		if (pop->ind[j].fitness > file_best_fitness[n_run]){		
			file_best_fitness[n_run] = pop->ind[j].fitness;
			for(int gene=0;gene<chrom_size;gene++)
				File_best_ind[n_run][gene]=pop->ind[j].chromosome[gene];
		}
	}

	pop->mean_fitness = pop->sum_fitness / popsize; 	// mean fitness in the population
	
	// Save data for fitness along the generations: only for the first run
	if (save_datagen_flag==1 && n_run==0){			
		if (gen<max_gen){		
			file_best_fitness_gen[gen]=pop->max_fitness;	
			if (mutLL_flag==1)
				file_n_edges_eVIG_gen[gen]=n_edges_eVIG;				
		}			
	}

	
}

