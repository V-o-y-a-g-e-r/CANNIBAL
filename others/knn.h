/******************************************************************************\
*				Class: K-Nearest Neighbors Algorithm					 	   *
\******************************************************************************/
#include <cmath>
#include <cstdlib>

#ifndef KNN_H
#define KNN_H


class KNN {
	private:
		int K;					// parameter K of KNN
		int n;					// number of examples in the dataset
		int at_list_size;		// number of selected attributes	
		int *at_list;			// list of selected attributes
		int *d;					// desired outputs of the training set
		double **X;				// inputs of the training set
	public:
		KNN(int K, double **X, int *d, int n, int *at_list, int at_list_size);		
		int modelOutput(double **X_testset, int i, int n_class);		// Output of the K-Nearest Neighbors Algorithm	
		double eucDist2(int j, double **X_testset, int i); 				// Euclidean Distance raised to 2	
};


/******************************************************************************\
*								Constructor									   *
\******************************************************************************/
KNN::KNN(int K, double **X, int *d, int n, int *at_list, int at_list_size){	

	this->K=K;							// parameter K of KNN
	this->n=n;							// number of examples of the training set	
	this->at_list_size=at_list_size;	// number of selected attributes
	this->at_list=at_list;				// list of selected attributes
	this->X=X;							// inputs of the training set	
	this->d=d;							// desired outputs of the training set

}


/******************************************************************************\
*				Euclidean Distance raised to 2						   		   *
\******************************************************************************/
double KNN::eucDist2(int j, double **X_testset, int i){
	double sum=0.0;

	for (int k=0;k<at_list_size;k++)
		sum=sum+pow( (X[j][at_list[k]]-X_testset[i][at_list[k]]), 2 );

	return sum;
}


/******************************************************************************\
*			Output of the K-Nearest Neighbors Algorithm				           *
\******************************************************************************/
int KNN::modelOutput(double **X_testset, int i, int n_class){
	int ind_min, max_c, *freq_class, *nearestNeig;
	double *dist_i, aux_d;
		
	freq_class = new int[n_class+1];			// vector with frequencies of each class
	dist_i = new double[n];						// vector with distances of training examples to the i-th example of the test set
	nearestNeig = new int[K];					// nearest neighbors
	
	// finding the distances to the i-th example of the test set
	for (int j=0;j<n;j++){
		dist_i[j]=eucDist2(j,X_testset,i);		// distance of the j-th example of the training set to the i-th example of the test set
	}
	
	// sorting K elements of the distance vector and finding K nearest neighbors
	for (int k=0;k<K;k++){	
		ind_min=k;		
		for (int j=k+1;j<n;j++){	
			if (dist_i[j]<dist_i[ind_min])
				ind_min=j;				
		}
		aux_d=dist_i[k];
		dist_i[k]=dist_i[ind_min];
		dist_i[ind_min]=aux_d;
		nearestNeig[k]=ind_min; 				// k-th nearest neighbors		
	}
	
	// finding the most frequent class
	for (int j=1;j<=n_class;j++)
		freq_class[j]=0;			
	for (int k=0;k<K;k++)
		freq_class[ d[nearestNeig[k]] ] +=1;
	max_c=1;
	for (int j=2;j<=n_classes;j++)
		if (freq_class[j]>freq_class[max_c])
			max_c=j;

	delete [] freq_class;
	delete [] dist_i;
	delete [] nearestNeig;
	
	return max_c;		// return most frequent class
}

#endif

