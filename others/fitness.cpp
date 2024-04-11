
/******************************************************************************\
*								 Feature Selection Problem				   	   *
\******************************************************************************/
#include "defs.h"
#include "knn.h"						// KNN class
#include "knn_reg.h"					// KNN - regression class

/******************************************************************************\
*			Train model and compute accuracy for the test set		           *
\******************************************************************************/
double compACC(int *at_list, int at_list_size){
	int n_hits=0, class_predict, K;

	
	if (classifier_type==1 || classifier_type==2){
		if (classifier_type==1){
			// Classifier 1: KNN with K=3
			K=3;
		}
		else{
			// Classifier 2: KNN with K=5
			K=5;
		}
	
		KNN *classifier_KNN = new KNN(K,X_trainset,d_trainset,n_examples_train, at_list, at_list_size);				// Create KNN 
		
		// Classification and number of hits
		for (int i=0;i<n_examples_test;i++){		
			class_predict = classifier_KNN->modelOutput(X_testset, i, n_classes);												// Classify i-th example of the test set with KNN
			if (class_predict==d_testset[i])
				n_hits++;				
		}	
		
		delete classifier_KNN;
	}
	else if (classifier_type==3){
		// Classifier 3:
		cout<<"Classifier "<<classifier_type<<"does not exist!"<<endl;
		exit(1);
	}
	else{
		cout<<"Classifier "<<classifier_type<<"does not exist!"<<endl;
		exit(1);
	}
					
	return ( (double) n_hits/n_examples_test);	
}


/******************************************************************************\
*			Train model and compute mean squared error (MSE) for the test set  *
\******************************************************************************/
double compMSE(int *at_list, int at_list_size){
	int K;
	double mse=0.0, output_predict;

	
	if (classifier_type==1 || classifier_type==2){
		if (classifier_type==1){
			// Classifier 1: KNN with K=3
			K=3;
		}
		else{
			// Classifier 2: KNN with K=5
			K=5;
		}
	
		KNN_REG *classifier_KNN = new KNN_REG(K,X_trainset,d_trainset_regression,n_examples_train, at_list, at_list_size);		// Create KNN 
		
		// compute MSE
		for (int i=0;i<n_examples_test;i++){		
			output_predict = classifier_KNN->modelOutput(X_testset, i);									// Output for the i-th example of the test set with KNN
			mse += pow(output_predict-d_testset_regression[i],2);				
		}	
		mse /= n_examples_test;		
		
		delete classifier_KNN;
	}
	else if (classifier_type==3){
		// Classifier 3:
		cout<<"Classifier "<<classifier_type<<"does not exist!"<<endl;
		exit(1);
	}
	else{
		cout<<"Classifier "<<classifier_type<<"does not exist!"<<endl;
		exit(1);
	}
				
	return ( 1.0 - mse );	
}


/******************************************************************************\
*			Fitness for the Feature Selection Problem				           *
\******************************************************************************/
double compFitFS(allele *ind){
	int  j=0, *at_list, at_list_size=0;
	double f, f1, f2;
	
	// computing the number of attributes
	for (int i=0;i<chrom_size;i++)
		at_list_size+=ind[i];		
	if (at_list_size==0)
		return (0.0);

	// setting the list of attributes
	at_list=aloc_vectori(at_list_size);
	for (int i=0;i<chrom_size;i++){
		if (ind[i]==1){
			at_list[j]=i;
			j++;
		}
	}
									
	// First term of the fitness
	if (prob_type==1){
		// Classification
		f1=compACC(at_list, at_list_size);   				// Train model and compute accuracy for the test set
	}
	else{
		// Classification
		f1=compMSE(at_list, at_list_size);   				// Train model and compute Mean squared error (MSE) for the test set
	}
			
	// Second term of the fitness
	f2=( (double) (chrom_size-at_list_size)) /chrom_size;  // term dependent on the number of selected attributes
		
	// compute fitness
	f = 0.98*f1 + 0.02*f2;									// weighted sum
	

	delete [] at_list;
	
	return f;
}



