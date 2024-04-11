/******************************************************************************\
*				Class: empirical Variable Interaction Graph				 	   *
\******************************************************************************/
#include <cmath>
#include <cstdlib>

#ifndef ESTVIG_H
#define ESTVIG_H


class estVIG {
	private:	
		typedef struct{
			int vertex;															// edge to vertex
			double weight;														// weight of the edge
		}  ListEntry;
		int N;																	// Number of decision variables
		int **W_count;															// Matrix for counting number of times each edge is visited (for computing averaged w)
	    double **W_sum;															// Matrix for the sum of w for each edge (for computing averaged w)	    
	    list<ListEntry> *eVIG; 													// Vector of lists eVIG - estimated Variable Interaction Graph (VIG) (eVIG[i]: indicates the variables that interact with the i-th variable )			
		void adjMatrix(void);													// Create Adjacency Matrix	
	public:
		int flag_Madj;															// flag for the adjacency matrix
		double **Madj;															// Adjacency Matrix
	  	estVIG(int N);
	    ~estVIG(void);
	    void print(void);														// Print eVIG
	    void save(char *prob_name, int classifier_type, int n_run, int type_GA); // Save eVIG
	    int addEdge(int a, int b, double w);									// Add edge (a,b) to the eVIG
};


/******************************************************************************\
*								Constructor									   *
\******************************************************************************/
estVIG::estVIG(int N){	
	
	// Parameters for eVIG
	this->N = N;	
	flag_Madj=0;				// indicates that the adjacency does not exist
				
	// Vector of lists eVIG - estimated Variable Interaction Graph (VIG) (eVIG[i]: indicates the variables that interact with the i-th variable )	
	eVIG = new list<ListEntry>[N]; 				
	
	// Matrices for computing the weights
	W_sum = new double*[N];
	for (int i=0;i<N;i++) 
		W_sum[i] = new double[N];
	W_count = new int*[N];
	for (int i=0;i<N;i++) 
		W_count[i] = new int[N];
	for (int i=0;i<N;i++){
		for (int j=0;j<N;j++){
			W_sum[i][j] = 0.0;
			W_count[i][j] = 0;
		}
	}	

}


/******************************************************************************\
*								 Destructor									   *
\******************************************************************************/
estVIG::~estVIG(void){
	
	for(int i=0;i<N;i++)
		delete [] W_count[i];
	delete [] W_count;
	for(int i=0;i<N;i++)
		delete [] W_sum[i];
	delete [] W_sum;
	
	delete [] eVIG; 
	
	if (flag_Madj==1){
		// desalloc adjacency matrix
		for(int i=0;i<N;i++) {
			delete [] Madj[i];
		}
		delete [] Madj;		
	}
		          
}  


/******************************************************************************\
*								Print Mk information						   *
\******************************************************************************/
void estVIG::print(void){

	cout<<"eVIG: variables "<<endl;
	for (int i=0;i<N;i++){
		cout<<" x_"<<i<<": ";
		for(list<ListEntry>::iterator j = eVIG[i].begin(); j != eVIG[i].end(); j++){
			cout<<"  x_"<<(*j).vertex;
			cout<<"("<<(*j).weight<<")";
		}
		cout<<endl;
	}
		
}


/******************************************************************************\
*			Create Adjacency Matrix from Adjacency Lists					   *
\******************************************************************************/
void estVIG::adjMatrix(void){

	// matrix allocation
	if (flag_Madj==0){
		Madj = new double*[N];
		for (int i=0;i<N;i++) {
			Madj[i] = new double[N];
		}
		flag_Madj=1;
	}

	// filling the elements of the matrix
	for (int i=0;i<N;i++)
		for (int j=0;j<N;j++)
			Madj[i][j]=0.0;
	for (int i=0;i<N;i++){
		for(list<ListEntry>::iterator jj = eVIG[i].begin(); jj != eVIG[i].end(); jj++) {
			Madj[i][(*jj).vertex]=(*jj).weight;
		}
	}
}


/******************************************************************************\
*								Save eVIG information														   *
\******************************************************************************/
void estVIG::save(char *prob_name, int classifier_type, int n_run, int type_GA){
	FILE *eVIG_file;
	char *name_p;
	char name[1000];	
		
    name_p = name;
	sprintf(name,"eVIG_%s_c%d_a%d_r%d.csv",prob_name,classifier_type,type_GA,n_run);
	if ((eVIG_file = fopen(name_p,"w"))==NULL) {
		puts("The file eVIG to be saved cannot be open \n");
		exit(1);
	}
			
	// Save as adjacency matrix
	if (flag_Madj==0)
		adjMatrix();			// create adjacency Matrix if it does not exist yet		
	for (int i=0;i<N;i++){
		for (int j=0;j<N;j++){
			if (j<N-1)
				fprintf(eVIG_file,"%1.5f, ",Madj[i][j]);
			else
				fprintf(eVIG_file,"%1.5f",Madj[i][j]);			
		}
		if (i<N-1)
			fprintf(eVIG_file,"\n");
	}

	
	fclose(eVIG_file);
		
}


/******************************************************************************\
* 				Add edge (a,b) to the eVIG with weight w					   *
\******************************************************************************/
int estVIG::addEdge(int a, int b, double w){
	int aux_flag=1;
	ListEntry aux_e;
	list<ListEntry>::iterator ii;
	
	// compute weight as the averaged w
	W_sum[a][b]+=w;
	if (W_count[a][b]>0)
		w=W_sum[a][b]/W_count[a][b];				
	W_count[a][b] += 1;
	W_sum[b][a]=W_sum[a][b];
	W_count[b][a]=W_count[a][b];
	
	// Check if link (a,b) is already in the eVIG	
	if ( !eVIG[a].empty() && !eVIG[b].empty() ){
		ii=eVIG[a].begin();
		while (ii != eVIG[a].end() && (*ii).vertex != b)					
			ii++;
		if (ii != eVIG[a].end())	
			aux_flag=0;
	}
	if (aux_flag==1){
		// Edge edge (a,b) does not exist yet: add edge with weight w
		aux_e.weight=w;
		aux_e.vertex=b;
		eVIG[a].push_back(aux_e); 
		aux_e.vertex=a;
		eVIG[b].push_back(aux_e); 	
	}
	else{ 
		// Edge edge (a,b) exists in list eVIG[a]: replace weight (if weight is higher than the stored weight)
		(*ii).weight=w;
		// do the same for in list eVIG[b], but first find the edge
		ii=eVIG[b].begin();
		while (ii != eVIG[b].end() && (*ii).vertex != a)					
			ii++;
		if (ii != eVIG[b].end())
			(*ii).weight=w;
	}
	
	return aux_flag;
}

#endif

