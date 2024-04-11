/******************************************************************************\
*				  				 Files Manipulation							 *
\******************************************************************************/

#include "defs.h"
#include<cstring>
#include<fstream>


/*******************************************************************\
*	 Read the problem instance  								    *
\*******************************************************************/
void read_problem(char *prob_name){
	char line[CHAR_LEN], * keywords,Delimiters[] = " :=\n\t\r\f\v";
	char name[CHAR_LEN];
	int n_examples;									// number of examples in the dataset
	int *d_dataset;									// desired outputs of the dataset - classification
	double *d_dataset_regression;					// desired outputs of the dataset - regression
	double perc_train=0.70;							// percentage for spliting training and test sets
	double  **X_dataset;							// inputs of the dataset

    // <problem name>: name of the instance (dataset), without extension. An example of the dataset format is given in file ionosphere.dat. 
	//  In the file with the dataset, the inputs are normalized between 0 and 1 and the labels of the classes are integers (starting at 1).
	//  It is also recommended to shuffle the examples. 
	
	sprintf(name,"%s.dat", prob_name);					// file is in the same directory
	
	ifstream fin(name);
	if(fin.fail()){
			cout<<"file name error"<<endl;
			exit(0);
	}
	
	while((fin.getline(line, CHAR_LEN-1))){
			if(!(keywords = strtok(line, Delimiters)))
	  			continue;
	  		if(!strcmp(keywords, "TYPE")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &prob_type)){
					cout<<"TYPE error"<<endl;
					exit(0);
	  			}
			}
			if(!strcmp(keywords, "N_ATTRIBUTES")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &chrom_size)){
					cout<<"N_ATTRIBUTES error"<<endl;
					exit(0);
	  			}
			}
			if(!strcmp(keywords, "N_EXAMPLES")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &n_examples)){
					cout<<"N_EXAMPLES"<<endl;
					exit(0);
	  			}
				X_dataset=aloc_matrixd (n_examples,chrom_size);
				if (prob_type==1)
					d_dataset=aloc_vectori (n_examples);
				else
					d_dataset_regression=aloc_vectord (n_examples);							
			}
			if(!strcmp(keywords, "N_CLASSES")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &n_classes)){
					cout<<"N_CLASSES"<<endl;
					exit(0);
	  			}	
			}
			else if(!strcmp(keywords, "DATASET")){
	  			if(n_examples>0){
	  				for(int i=0; i<n_examples; i++){	  				
						for(int j=0; j<chrom_size; j++)						
							fin>>X_dataset[i][j];
						if (prob_type==1)						
							fin>>d_dataset[i];
						else 
							fin>>d_dataset_regression[i];
					}
	    		}
			}
	}
	fin.close();		
	
	// Splitting the dataset in two datasets (training and testing)
	n_examples_train=perc_train*n_examples;					// number of examples in training set
	n_examples_test=n_examples-n_examples_train;			// number of examples in test set
	X_trainset=aloc_matrixd (n_examples_train,chrom_size);
	if (prob_type==1)
		d_trainset=aloc_vectori (n_examples_train);
	else
		d_trainset_regression=aloc_vectord (n_examples_train);
	X_testset=aloc_matrixd (n_examples_test,chrom_size);
	if (prob_type==1)
		d_testset=aloc_vectori (n_examples_test);
	else
		d_testset_regression=aloc_vectord (n_examples_test);
	for(int i=0; i<n_examples; i++){
		if (i<n_examples_train){
			for(int j=0; j<chrom_size; j++)
				X_trainset[i][j]=X_dataset[i][j];
			if (prob_type==1)
				d_trainset[i]=d_dataset[i];
			else	
				d_trainset_regression[i]=d_dataset_regression[i];
		}
		else{
			for(int j=0; j<chrom_size; j++)
				X_testset[i-n_examples_train][j]=X_dataset[i][j];
			if (prob_type==1)
				d_testset[i-n_examples_train]=d_dataset[i];	
		    else
		    	d_testset_regression[i-n_examples_train]=d_dataset_regression[i];	
		}
		
	}
	
	desaloc_matrixd(X_dataset,n_examples);
	if (prob_type==1)
		delete [] d_dataset;
	else 
		delete [] d_dataset_regression;
		
}


/******************************************************************************\
* 					Save  : end of the simulation						   	   *
\******************************************************************************/
void file_output(char *prob_name, int total_runs)
{
	char *name_p;
	char name[CHAR_LEN];
	FILE *Bestfit_file, *Bestind_file, *Time_file, *Gen_file;

    name_p = name;

  	// Best fitness in each generation for run 0
  	if (save_datagen_flag==1){
  		FILE *Bfg_file;
		sprintf(name,"bfg_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
		if ((Bfg_file = fopen(name_p,"w"))==NULL) {
			puts("The file bfg to be saved cannot be open \n");
			exit(1);
		}
		for (int i=0;i<max_gen;i++) {
			fprintf(Bfg_file,"%.5f ",file_best_fitness_gen[i]);
		}
		fclose(Bfg_file);
		if (mutLL_flag==1){
			FILE *Nedges_gen_file;
			sprintf(name,"nedgesgen_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
			if ((Nedges_gen_file = fopen(name_p,"w"))==NULL) {
				puts("The file nedgesgen to be saved cannot be open \n");
				exit(1);
			}
			for (int i=0;i<max_gen;i++) {
				fprintf(Nedges_gen_file,"%.1f ",file_n_edges_eVIG_gen[i]);
			}
			fclose(Nedges_gen_file);			
		}
	}
	if (mutLL_flag==1){
		// Mean number of edges of the eVIG
		FILE *Nedges_file;
		sprintf(name,"nedges_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
		if ((Nedges_file = fopen(name_p,"w"))==NULL) {
			puts("The file nedges to be saved cannot be open \n");
			exit(1);
		}
		for (int i=0;i<total_runs;i++) {					
			fprintf(Nedges_file,"%.1f ",file_n_edges_eVIG[i]);
		}		
		fclose(Nedges_file);
	}

    // Best fitness 
	sprintf(name,"bfi_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
	if ((Bestfit_file = fopen(name_p,"w"))==NULL) {
		puts("The file bfi to be saved cannot be open \n");
		exit(1);
	}
	for (int i=0;i<total_runs;i++) {
		fprintf(Bestfit_file,"%.14f ",file_best_fitness[i]);
	}
	fclose(Bestfit_file);
		
	 // Best individuals
	sprintf(name,"bind_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
	if ((Bestind_file = fopen(name_p,"w"))==NULL) {
		puts("The file bind to be saved cannot be open \n");
		exit(1);
	}
	for (int i=0;i<total_runs;i++) {
		for (int gene=0;gene<chrom_size;gene++)
			fprintf(Bestind_file,"%d ",File_best_ind[i][gene]);
		fprintf(Bestind_file,"\n");
	}
	fclose(Bestind_file);
	
  	// Time for each run
	sprintf(name,"time_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
	if ((Time_file = fopen(name_p,"w"))==NULL) {
		puts("The file time to be saved cannot be open \n");
		exit(1);
	}
	for (int i=0;i<total_runs;i++) {
		fprintf(Time_file,"%.2f ",time_run[i]);
	}
	fclose(Time_file);
	
	// Number of generations for each run
	sprintf(name,"gen_%s_c%d_a%d.dat",prob_name,classifier_type,mutLL_flag);
	if ((Gen_file = fopen(name_p,"w"))==NULL) {
		puts("The file gen to be saved cannot be open \n");
		exit(1);
	}
	for (int i=0;i<total_runs;i++) {
		fprintf(Gen_file,"%d ",file_gen[i]);
	}
	fclose(Gen_file);				
	
}
