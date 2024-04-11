/******************************************************************************\
*				Class: K-Nearest Neighbors Algorithm for Regresssion     	   *
\******************************************************************************/
#include <cmath>
#include <cstdlib>

#ifndef KNN_REG_H
#define KNN_REG_H


class KNN_REG {
	private:
		int K;					// parameter K of KNN
		int n;					// number of examples in the dataset
		int at_list_size;		// number of selected attributes	
		int *at_list;			// list of selected attributes
		double *d;				// desired outputs of the training set
		double **X;				// inputs of the training set
	public:
		KNN_REG(int K, double **X, double *d, int n, int *at_list, int at_list_size);		
		double modelOutput(double **X_testset, int i);					// Output of the K-Nearest Neighbors Algorithm	
		double eucDist2(int j, double **X_testset, int i); 				// Euclidean Distance raised to 2	
};


/******************************************************************************\
*								Constructor									   *
\******************************************************************************/
KNN_REG::KNN_REG(int K, double **X, double *d, int n, int *at_list, int at_list_size){	

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
double KNN_REG::eucDist2(int j, double **X_testset, int i){
	double sum=0.0;

	for (int k=0;k<at_list_size;k++)
		sum=sum+pow( (X[j][at_list[k]]-X_testset[i][at_list[k]]), 2 );

	return sum;
}


/******************************************************************************\
*			Output of the K-Nearest Neighbors Algorithm				           *
\******************************************************************************/
double KNN_REG::modelOutput(double **X_testset, int i){
	int ind_min, *nearestNeig;
	double *dist_i, aux_d, y=0.0;
		
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
	
	// computing the output			
	for (int k=0;k<K;k++){
		y += d[nearestNeig[k]];
	}
	y /= K;
		

	delete [] dist_i;
	delete [] nearestNeig;
	
	return y;		// return output
}

#endif

