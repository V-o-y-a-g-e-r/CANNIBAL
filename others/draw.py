import pandas as pd
import networkx as nx

input_data = pd.read_csv('1000 samples/eVIG_dataset_c1_a1_r1.csv', sep=",", header=None)
G = nx.DiGraph(input_data.values[:10, :10])
input_data.shape
nx.draw(G)
