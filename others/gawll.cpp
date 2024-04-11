/*******************************************************************************\
*  	Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 		  	   	*
*																				*
* 	Genetic Algorithm with Linkage Learning (GAwLL)								*
* 	for the Wrapper Feature Selection Problem 									*
*						 														*
* 	Copyright (C) 2023  Renato Tinos <rtinos@ffclrp.usp.br>						*
* 						 														*
* 	Reference: Tinos, R.; Przewozniczek, M.; Whitley, D. & Chicano, F. (2023). 	*                    
* 				"Genetic Algorithm with Linkage Learning",		 				*
*				Submitted to GECCO'2023.										*
*																				*
* 	gawll_fs is free software: you can redistribute it and/or modify it 		*
* 		under the terms of the GNU General Public License as published by the	*
* 		Free Software Foundation, either version 3 of the License, or			*
* 		(at your option) any later version.						 				*
* 						 														*
* 	gawll_fs is distributed in the hope that it will be useful, but				*
* 		WITHOUT ANY WARRANTY; without even the implied warranty of				*
* 		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.					*
* 		See the GNU General Public License for more details.					*
* 																				*
* 	You should have received a copy of the GNU General Public License along		*
* 		with this program.  If not, see <http://www.gnu.org/licenses/>.			*		
\*******************************************************************************/

#include <time.h>
#include "defs.h"
#include "estVIG.h"						// estimated VIG class


/******************************************************************************\
*				  	Print population					 			 			*
\******************************************************************************/
void print_data(population *pop, int n_run){

	cout <<"Generation:"<< gen << ", run: "<<n_run<<endl;
	cout <<"Best individual:"<< pop->best_individual << endl;
	cout <<"Fitness of the best individual:"<< pop->max_fitness << endl;
	cout <<"Mean fitness: "<< pop->mean_fitness << endl;	
	for (int i=0;i<popsize ;i++) {	
		cout <<"("<< pop->ind[i].fitness<<") " ;
		for (int gene=0;gene<chrom_size ;gene++) 
			cout << pop->ind[i].chromosome[gene]<<" ";
		cout << endl;
	}
	
}


/******************************************************************************\
*								Fitness  Computation    					   *
\******************************************************************************/
double compFitness(allele *x){
	double Fitness;

	Fitness = compFitFS(x);			// fitness function for Feature selection Problem
		
	return Fitness;
}


/******************************************************************************\
*					Local Search: First Improvement							   *
\******************************************************************************/
double LS(allele *x, double f){
	int i, j=0, ni=0, nr=0, *R;
	double df, fit_aux;

	// Randomly defining the order of the search
	R=aloc_vectori(chrom_size);							// random sequence for visiting variables
	rand_perm_size(vsort_aux, R, chrom_size, chrom_size);
	
	// Local Search	
	while(ni<chrom_size){
		i=R[j];
		// Flipping the i-th bit
		if (x[i]==0)
			x[i]=1;
		else
			x[i]=0;
		
		// Computing the difference in the fitness
		fit_aux=compFitness(x);			// fitness computation
		df=fit_aux-f;
		ni++;
		
		// Cheking improvement
		if (df<EPS2){	
			//  No Improvement: Reversing the flip
			if (x[i]==0)
				x[i]=1;
			else
				x[i]=0;
		}
		else{
			// Improvement
			f=fit_aux;
			ni=0;
			nr++;
			if (nr>5*chrom_size)
				ni=chrom_size;				// a limit for the number of repetitions (improvements) is here defined					
		}
		j++;
		if (j==chrom_size)
			j=0;										 	
	}
	
	delete [] R;
	
	return f;

}


/***********************************************************************************************************\
*								 Crossover	 													     		*
* Crossover type: 1-2PointX; 2-UX																		    *
\***********************************************************************************************************/
void crossover(allele *parent1, allele *parent2, allele *offspring1, allele *offspring2){	
	int type_crossover=2;		// here, the uniform crossover is used

	if (type_crossover==1){
		// 2-point Crossover
		Point2X(parent1,parent2,offspring1,offspring2);
	}
	else if (type_crossover==2){
		// Uniform Crossover
		UX(parent1,parent2,offspring1,offspring2);	
	}	
	else{
		cout<<"Crossover type does not exist!"<<endl;
		exit(1);
	}
	
}


/******************************************************************************\
*					Mutation with Linkage Learning							   *
\******************************************************************************/
void mutationLL(int parent, int j, estVIG *eVIG_instance){
	int g, h;
	double fx, fxg, fxh, fxhg, df;
	
	for (int gene=0;gene<chrom_size;gene++){
		popnew.ind[j].chromosome[gene]=popold.ind[parent].chromosome[gene]; 		// individal xg
		popnew.ind[j+1].chromosome[gene]=popold.ind[parent].chromosome[gene];		// individual xh
		popnew.ind[j+2].chromosome[gene]=popold.ind[parent].chromosome[gene];		// individual xgh
	}
	
	// Individual x: parent
	fx=popold.ind[parent].fitness;
			
	// Individual xg: individual x with mutation in x[g]		
	g=random_int(0,chrom_size-1);
	if (popnew.ind[j].chromosome[g]==0)
		popnew.ind[j].chromosome[g]=1;
	else
		popnew.ind[j].chromosome[g]=0;
	popnew.ind[j].fitness=compFitness(popnew.ind[j].chromosome);				
	fxg=popnew.ind[j].fitness;
	
	// Individual xh: individual x with mutation in x[h]
	do{		
		h=random_int(0,chrom_size-1);
	} while (h==g);
	if (popnew.ind[j+1].chromosome[h]==0)
		popnew.ind[j+1].chromosome[h]=1;
	else
		popnew.ind[j+1].chromosome[h]=0;
	popnew.ind[j+1].fitness=compFitness(popnew.ind[j+1].chromosome);				
	fxh=popnew.ind[j+1].fitness;
	
	// Individual xhg: individual x with mutation in x[h] and x[g]
	popnew.ind[j+2].chromosome[g]=popnew.ind[j].chromosome[g];
	popnew.ind[j+2].chromosome[h]=popnew.ind[j+1].chromosome[h];
	popnew.ind[j+2].fitness=compFitness(popnew.ind[j+2].chromosome);				
	fxhg=popnew.ind[j+2].fitness;
	
	df=fabs(fxhg-fxh-fxg+fx);		
	if ( df>EPS2 ){
		// df changed because variables g and h interact
		if (eVIG_instance->addEdge(g,h,df)){			// adding edge (g,h) to eVIG (if the edges does not exist)
			n_edges_eVIG=n_edges_eVIG+1;				// update number of edges in eVIG
		}
	}	
	
}


/******************************************************************************\
*								 Generation of the GA with LL				   *
\******************************************************************************/
void generation_LL(int n_run, estVIG *eVIG_instance){
	int j=0, j_best, parent1, parent2, nc;
	
	nc=p_cross*popsize;			// number of individuals for standard crossover and mutation
	
	// Elitism (for one individual: with index j=0)
	for (int gene=0;gene<chrom_size;gene++)
		popnew.ind[j].chromosome[gene]=popold.ind[popold.best_individual].chromosome[gene];
	popnew.ind[j].fitness=popold.ind[popold.best_individual].fitness;
	
	j++;
	while ( j < popsize){		
		parent1=selection( &popold );			// Selection of the first parent 
		
		if ( j<nc){
			// Standard crossover and mutation: generate 2 individuals each time
			parent2=selection( &popold );		// Selection of the second parent 				
			crossover( popold.ind[parent1].chromosome , popold.ind[parent2].chromosome,  popnew.ind[j].chromosome, popnew.ind[j+1].chromosome);										
			mutation(popnew.ind[j].chromosome);
			mutation(popnew.ind[j+1].chromosome);
			// Evaluation
			popnew.ind[j].fitness=compFitness(popnew.ind[j].chromosome);
			popnew.ind[j+1].fitness=compFitness(popnew.ind[j+1].chromosome);			
			if (popnew.ind[j].fitness>popnew.ind[j+1].fitness)
				j_best=j;
			else
				j_best=j+1;
			
			j+=2;
		}
		else if( j<popsize-2 ){
			// Mutation with Linkage Learning: generate 3 individuals each time			
			mutationLL(parent1, j, eVIG_instance);	
			if (popnew.ind[j].fitness>popnew.ind[j+1].fitness){
				if (popnew.ind[j].fitness>popnew.ind[j+2].fitness)
					j_best=j;	
				else
					j_best=j+2;	
			}
			else{
					if (popnew.ind[j+1].fitness>popnew.ind[j+2].fitness)
					j_best=j+1;	
				else
					j_best=j+2;				
			}			
			j+=3;
		}				
		else {			
			// Standard mutation: generate one individual each time
			for (int gene=0;gene<chrom_size;gene++)
				popnew.ind[j].chromosome[gene]=popold.ind[parent1].chromosome[gene];
			mutation(popnew.ind[j].chromosome);	
			popnew.ind[j].fitness=compFitness(popnew.ind[j].chromosome);
			j_best=j;	
			j++;		
		}
				
		checkBestInd(&popnew, j_best, n_run);			// check if best offspring is better than best stored individual 		
	} 
			
}


/******************************************************************************\
*								 Generation of the Standard GA 				   *
\******************************************************************************/
void generation(int n_run ){
	int j=0, j_best, parent1, parent2;
	
	// Elitism (for one individual: with index j=0)
	for (int gene=0;gene<chrom_size;gene++)
		popnew.ind[j].chromosome[gene]=popold.ind[popold.best_individual].chromosome[gene];
	popnew.ind[j].fitness=popold.ind[popold.best_individual].fitness;
	
	j++;
	while ( j < popsize){		
		parent1=selection( &popold );			// Selection of the first parent 
		
		if ( j<popsize-1 ){
			// Standard crossover and mutation: generate 2 individuals each time
			parent2=selection( &popold );		// Selection of the second parent 	
			if (random_dou()<p_cross) {			
				crossover( popold.ind[parent1].chromosome , popold.ind[parent2].chromosome,  popnew.ind[j].chromosome, popnew.ind[j+1].chromosome);	
			}
			else{
				for (int gene=0;gene<chrom_size;gene++){
					popnew.ind[j].chromosome[gene]=popold.ind[parent1].chromosome[gene];
					popnew.ind[j+1].chromosome[gene]=popold.ind[parent2].chromosome[gene];
				}
			}									
			mutation(popnew.ind[j].chromosome);
			mutation(popnew.ind[j+1].chromosome);
			// Evaluation
			popnew.ind[j].fitness=compFitness(popnew.ind[j].chromosome);
			popnew.ind[j+1].fitness=compFitness(popnew.ind[j+1].chromosome);			
			if (popnew.ind[j].fitness>popnew.ind[j+1].fitness)
				j_best=j;
			else
				j_best=j+1;
			
			j+=2;
		}					
		else {			
			// Standard mutation: generate one individual each time
			for (int gene=0;gene<chrom_size;gene++)
				popnew.ind[j].chromosome[gene]=popold.ind[parent1].chromosome[gene];
			mutation(popnew.ind[j].chromosome);	
			popnew.ind[j].fitness=compFitness(popnew.ind[j].chromosome);
			j_best=j;	
			j++;		
		}
				
		checkBestInd(&popnew, j_best, n_run);			// check if best offspring is better than best stored individual 		
	} 
			
}

/******************************************************************************\
*				  	Random individual					 				 	   *
\******************************************************************************/
void randomInd(int num_ind){
	
	for (int gene=0;gene<chrom_size;gene++) 
     	popold.ind[num_ind].chromosome[gene] = random_int(0,1);

    popold.ind[num_ind].fitness = compFitness(popold.ind[num_ind].chromosome);											
	
}


/******************************************************************************\
*				  	Initiate Population 					 				 *
\******************************************************************************/
void initiatePop(int n_run){
				
	// Dynamic allocation: populations
	popold.ind = aloc_vectorind(popsize);
	popnew.ind = aloc_vectorind(popsize);

	for (int num_ind=0;num_ind<popsize;num_ind++){
		// Dynamic allocation: chromosomes	
		popold.ind[num_ind].chromosome = aloc_vectori(chrom_size);
		popnew.ind[num_ind].chromosome = aloc_vectori(chrom_size);

		// Random Initialization
		randomInd(num_ind);	 	
      	
      	if (LS_flag==1){
      		// Applying Local Search 
			popold.ind[num_ind].fitness = LS(popold.ind[num_ind].chromosome,popold.ind[num_ind].fitness); 		// local search: first improvement (for Mk landscapes)				
		}
	}
	file_best_fitness[n_run]=popold.ind[0].fitness;
	for(int i=0;i<chrom_size;i++)
     	File_best_ind[n_run][i]=popold.ind[0].chromosome[i];
	statistics(&popold, n_run);
	//print_data(&popold, n_run);
	
}


/******************************************************************************\
*				  	Copy Population							 			 	   *
\******************************************************************************/
void copy_pop( void ){
		
	for (int ind=0;ind<popsize;ind++) {	
		popold.ind[ind].fitness=popnew.ind[ind].fitness;
		for (int gene=0;gene<chrom_size;gene++) 
			popold.ind[ind].chromosome[gene]=popnew.ind[ind].chromosome[gene];
	}
	
}


/******************************************************************************\
*				  	Run of the GA 			 								   *
\******************************************************************************/
void ga(char *prob_name, int n_run ){
	int j;
	int last_change_gen;
	double time_aux, max_time; 
	double last_change_fitness;
	clock_t time_start;
				
	// Initialization
	estVIG *eVIG_instance = new estVIG(chrom_size);				// from class estVIG (estVIG.h)
	max_time=chrom_size*n_examples_train/10.0;	
	time_start=clock();	
	if (mutLL_flag==1){	
		n_edges_eVIG=0;
	}
	gen=0;
	initiatePop(n_run);											// initiating population
	last_change_fitness=popold.max_fitness;
	last_change_gen=gen;
	
	do {
		gen++; 													// generation index
		if ( (popold.max_fitness-last_change_fitness) > EPS2 ){
			last_change_fitness=popold.max_fitness;
			last_change_gen=gen;	
		}
		if (  (gen-last_change_gen) > tau_reset ){
			//Reseting part of the population and applying local search
			
			// Elitism
			for (int gene=0;gene<chrom_size;gene++)
				popold.ind[0].chromosome[gene]=popold.ind[popold.best_individual].chromosome[gene];
			popold.ind[0].fitness=popold.ind[popold.best_individual].fitness;
			popold.best_individual=0; 
			// Reseting part of the population
			j=(int) (resetpop_rate*popsize) + 1;
			if (j>popsize)
				j=popsize;
			for (int num_ind=1;num_ind<j;num_ind++)
				randomInd(num_ind);			
			if (LS_flag==1){	
				// Applying local search
				for (int num_ind=0;num_ind<popsize;num_ind++){
					popold.ind[num_ind].fitness = LS(popold.ind[num_ind].chromosome,popold.ind[num_ind].fitness); 				// local search: first improvement (for Mk landscapes)				
				}
			}
			statistics(&popold,n_run);
			last_change_fitness=popold.max_fitness;
			last_change_gen=gen;	
		}
		
		if (mutLL_flag==1)
			generation_LL(n_run,eVIG_instance);						// GA with LL
		else
			generation(n_run);										// standard GA
			
		copy_pop();													// popold=popnew
		
		statistics(&popold,n_run);	
		//print_data(&popold,n_run);
					
		time_aux = ( (double) ( clock() - time_start ) ) / ( (double) CLOCKS_PER_SEC);

	}while ( time_aux < max_time );									// for experiments with fixed time
	//}while ( gen < max_gen );    									// for experiments with fixed number of iterations
		
	// Data to be saved
	time_run[n_run]=time_aux;
	file_gen[n_run]=gen;		
	if (mutLL_flag==1){
		file_n_edges_eVIG[n_run]=n_edges_eVIG;
		//eVIG_instance->print();									// print empirical Variable Interaction Graph (VIG)	
		eVIG_instance->save(prob_name, classifier_type, n_run, mutLL_flag);		// save VIG
	}
	
	// Deleting population
	for (int num_ind=0;num_ind<popsize;num_ind++){
		delete [] popold.ind[num_ind].chromosome;
		delete [] popnew.ind[num_ind].chromosome;
	}
	delete [] popold.ind;
	delete [] popnew.ind;
	
	delete eVIG_instance;

}


/******************************************************************************\
*				  	Main													   *
\******************************************************************************/
int main(int argc , char *argv[])
{
	int total_runs, n_run=0;
	char *prob_name;
		
	// Arguments
	if( argc < 4) {
		cout<<"Insufficient number of arguments!"<<endl;
		cout<<"Call: gawll_fs <problem name - without extension> <classifier> <GA_type>"<<endl;
		exit(1);
	}
	else{
		prob_name=argv[1];
		// load dataset
		read_problem(prob_name);
		classifier_type=atoi(argv[2]);
		mutLL_flag=atoi(argv[3]);
		if ( classifier_type<1 || classifier_type>3 || mutLL_flag<0 || mutLL_flag>1 ) {
			cout<<"Incorrect arguments!"<<endl;
			cout<<"Call: gawll_fs <problem name - without extension> <classifier> <GA_type> (0-Standard GA; 1-GAwLL)"<<endl;
			exit(1);
		}
	}	
	
	// Parameters
	p_mut=1.0/chrom_size;												// mutation rate
	total_runs=n_runs_per_instance;										// number of runs
	max_gen=2000;														// maximum number of generations (when stop criterion is time, it is used for saving some statistics data)	
						
	// Allocation of vectors and matrices
	File_best_ind=aloc_matrixi(total_runs,chrom_size);
	file_best_fitness_gen=aloc_vectord(max_gen);
	file_best_fitness=aloc_vectord(total_runs);
	file_n_edges_eVIG=aloc_vectord(total_runs);
	file_n_edges_eVIG_gen=aloc_vectord(max_gen);
	file_gen=aloc_vectori(total_runs);
	time_run=aloc_vectord(total_runs);
	vsort_aux=aloc_vectori(chrom_size);									// Auxiliar sorted vector of integers (used in different methods)
	for (int i=0;i<chrom_size;i++)
		vsort_aux[i]=i;
	
	cout << "\n ***** Hybrid Genetic Algorithm ****" << endl;
	cout << "Feature Selection, Dataset: "<<prob_name << endl;
	cout << "GA Model (0-Standard GA; 1-GAwLL)="<<mutLL_flag<< endl;	
	
	for (n_run=0;n_run<total_runs;n_run++) {	
		srand(n_run+1);													// random seed   		
		cout <<"Run:"<< n_run << endl;
		ga(prob_name, n_run);					    					// run GA
	}	
		
	file_output(prob_name, total_runs);									// save data

	// Desallocation of vectors and matrices	
	desaloc_matrixi (File_best_ind,total_runs);
	delete [] d_trainset;
	delete [] d_testset;
	delete [] time_run;
	delete [] file_gen;
	delete [] file_best_fitness;
	delete [] file_best_fitness_gen;
	delete [] file_n_edges_eVIG;
	delete [] file_n_edges_eVIG_gen;
	delete [] vsort_aux;
		
	return 0;
}
