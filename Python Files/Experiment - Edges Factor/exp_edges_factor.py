#!/usr/bin/env python
# coding: utf-8

#### Similarity Distribution Over Random Graphs of Varying Size

# Import necessary libraries
import sys
sys.path.append('../')
from similarity_distribution import *


### Calculating Similarity Distribution Over Edges Factor

def getEdgesFactorDistribution(num_graphs, num_nodes, start_edges_factor, end_edges_factor, step, column_list, metrics):
    cols = pd.MultiIndex.from_product([column_list, metrics])
    edge_factor_dist = pd.DataFrame(index = [["Nodes To Edges Factor"], ["Number of Edges Per Graph"], ["Number of Graphs"]], columns = cols)

    for nodes_to_edges_factor in np.arange(start_edges_factor, end_edges_factor, step):
        nodes_to_edges_factor = round(nodes_to_edges_factor, 1) # round edges factor to one decimal place
        num_edges = int(num_nodes * nodes_to_edges_factor)

        graph_similarity_list = getGraphSimilarityList(num_graphs, num_nodes, num_edges)
        graph_distribution = getCondensedSimilarityDistribution(graph_similarity_list, column_list, metrics)
        
        new_row = getNewRow(graph_distribution, column_list)
        edge_factor_dist.loc[(nodes_to_edges_factor, num_edges, num_graphs),:] = new_row

    return edge_factor_dist


## Helper Functions

def getNewRow(graph_distribution, column_list): 
    new_row = []
    
    for column in column_list:
        similarities = graph_distribution[column]["mean"]
        new_row.append(similarities.mean())
        new_row.append(similarities.median())
        new_row.append(similarities.std())

    return tuple(new_row)



### Running The Script
## Get Parameters and Constants
parameters = pd.read_csv("parameters-edges-factor.csv")

COLUMN_LIST = parameters["Column List"].to_list()
METRICS = [metric for metric in parameters["Metrics"].to_list() if str(metric) != 'nan']

NUM_NODES = int(parameters["Number of Nodes"][0])
NUM_GRAPHS = int(parameters["Number of Graphs"][0])
START_EDGES_FACTOR = parameters["Starting Nodes To Edges Factor"][0]
END_EDGES_FACTOR = parameters["Ending Nodes To Edges Factor"][0]
STEP = parameters["Step"][0]

## Run The Experiment
print("Beginning the experiment...")
print("Starting with", str(START_EDGES_FACTOR), "nodes-to-edges-factor, ending with", str(END_EDGES_FACTOR), "with a step of", str(STEP) + ".")
print(str(NUM_GRAPHS), "graphs will be made for each node-to-edges-factor size, and each graph will have", str(NUM_NODES), "nodes.")
edge_factor_dist = getEdgesFactorDistribution(NUM_GRAPHS, NUM_NODES, START_EDGES_FACTOR, END_EDGES_FACTOR, STEP, COLUMN_LIST, METRICS)

## Exporting Data
file_name = "Data/edge-factor-dist-" + str(NUM_GRAPHS) + "-" + str(NUM_NODES) + "-" + str(START_EDGES_FACTOR) + "-" + str(END_EDGES_FACTOR) + "-" + str(STEP) + ".csv"
edge_factor_dist.to_csv(file_name, index=True)
print("The program has finished running, and the experiment is complete. The data is saved to", file_name + ".")
