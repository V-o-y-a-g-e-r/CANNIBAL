gawll_fs : aux_functions.o file_man.o fitness.o global.o gawll.o selection.o statistics.o transformation.o
	g++ -Wall aux_functions.o file_man.o fitness.o global.o gawll.o selection.o statistics.o transformation.o -o gawll_fs

aux_functions.o : aux_functions.cpp	
	g++ -Wall -o aux_functions.o -c aux_functions.cpp

file_man.o : file_man.cpp	
	g++ -Wall -o file_man.o -c file_man.cpp

fitness.o : fitness.cpp	
	g++ -Wall -o fitness.o -c fitness.cpp

gawll.o : gawll.cpp	
	g++ -Wall -o gawll.o -c gawll.cpp

global.o : global.cpp	
	g++ -Wall -o global.o -c global.cpp

selection.o : selection.cpp	
	g++ -Wall -o selection.o -c selection.cpp

statistics.o : statistics.cpp	
	g++ -Wall -o statistics.o -c statistics.cpp

transformation.o : transformation.cpp	
	g++ -Wall -o transformation.o -c transformation.cpp

