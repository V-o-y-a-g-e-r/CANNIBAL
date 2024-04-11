*** Genetic Algorithm with Linkage Learning (GAwLL) for the Wrapper Feature Selection Problem  ***

Description: This is the source code for running GAwLL in the feature selection problems described in the paper:

Reference:  Tinos, R.; Przewozniczek, M.; Whitley, D. & Chicano, F. (2023). "Genetic Algorithm with Linkage Learning", Submitted to GECCO'2023.    

It also includes the code for running the standard Genetic Algorithm described in the paper.

Contact: Renato Tinos <rtinos@ffclrp.usp.br>

Running the code: ./gawll_fs <problem name> <classifier> <GA_type>

<problem name>: name of the instance (dataset), without extension. An example of the dataset format is given in file ionosphere.dat.
 In the file with the dataset, the inputs are normalized between 0 and 1. The outputs for classification are labels of the classes 
 indicated by integers (starting at 1). The outputs for regression are normalized between 0 and 1. It is also recommended to shuffle the
  examples. The first lines of the file indicate the name of the dataset, the type of the machine learning problem (1-classification; 2-regression), 
  the number of attributes, the number of examples, and the number of classes (only for classification; in regression, only one output is considered). 
  Example for the ionosphere dataset:
MODEL: IONOSPHERE 
TYPE: 1 
N_ATTRIBUTES: 34 
N_EXAMPLES: 351 
N_CLASSES: 2 

<classifier>: machine learning model (here, only KNN is used). 1: KNN with K=3; 2: KNN with K=5.

<GA_type>: genetic algorithm (GA) model. 0: standard GA; 1: GA with linkage learning (GAwLL)

Example for running the code for: ionosphere (dataset ionosphere) 1 (KNN with K=3) 1 (gawll)

make

./gawll_fs ionosphere 1 1 

	
Observation 1: file global.cpp contains the parameters of the GA (examples: number of runs, population size, and crossover rate).

Observation 2: gawll generates 4 main files
 
- "bfi_%s_c%d_a%d.dat",prob_name,classifier_type,GA_type: best fitness found in each run
	
- "bind_%s_c%d_a%d.dat",prob_name,classifier_type,GA_type: best individuals found in each run

- "time_%s_c%d_a%d.dat",prob_name,classifier_type,GA_type: time for each runx

- "gen_%s_c%d_a%d.dat",prob_name,classifier_type,GA_type: number of generations for each run

- "nedges_%s_c%d_a%d.dat",prob_name,classifier_type,GA_type: mean number of edges of the empirical VIG for each run (only for GAwLL)

- "eVIG_%s_c%d_a%d_r%d.csv",prob_name,classifier_type,GA_type,n_run: save empirical VIG (only for GAwLL) for run n_run

