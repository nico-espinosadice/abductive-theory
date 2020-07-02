#!/usr/bin/env python
# coding: utf-8

#### Similarity Distribution Over Random Graphs of Varying Size

# Import necessary libraries
import sys
sys.path.append('../')
from similarity_distribution import *


### Calculating Similarity Distribution Over Graph Sizes

def getGraphSizeDistribution(num_graphs, start_num_nodes, end_num_nodes, step, nodes_to_edges_factor, column_list, metrics):    
    cols = pd.MultiIndex.from_product([column_list, metrics])
    graph_size_dist = pd.DataFrame(index = [["Number of Nodes Per Graph"], ["Number of Graphs"]], columns = cols)

    for num_nodes in range(start_num_nodes, end_num_nodes, step):
        num_edges = int(num_nodes * nodes_to_edges_factor)
        
        graph_similarity_list = getGraphSimilarityList(num_graphs, num_nodes, num_edges)
        graph_distribution = getCondensedSimilarityDistribution(graph_similarity_list, column_list, metrics)
        
        new_row = getNewRow(graph_distribution, column_list)
        graph_size_dist.loc[(num_nodes, num_graphs),:] = new_row

    return graph_size_dist


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
parameters = pd.read_csv("parameters-graph-size.csv")

COLUMN_LIST = parameters["Column List"].to_list()
METRICS = [metric for metric in parameters["Metrics"].to_list() if str(metric) != 'nan']

NODES_TO_EDGES_FACTOR = int(parameters["Nodes To Edges Factor"][0])
NUM_GRAPHS = int(parameters["Number of Graphs"][0])
START_NUM_NODES = int(parameters["Starting Number of Nodes"][0])
END_NUM_NODES = int(parameters["Ending Number of Nodes"][0])
STEP = int(parameters["Step"][0])


## Run The Experiment
print("Beginning the experiment...")
print("Making graphs starting with", str(START_NUM_NODES), "and going until", str(END_NUM_NODES), "with a step of", str(STEP) + ".")
print(str(NUM_GRAPHS), "will be made for each graph size, and each graph will have", str(NODES_TO_EDGES_FACTOR) + "x the number of edges as it does nodes.")
graph_size_dist = getGraphSizeDistribution(NUM_GRAPHS, START_NUM_NODES, END_NUM_NODES, STEP, NODES_TO_EDGES_FACTOR, COLUMN_LIST, METRICS)

## Exporting Data
file_name = "Data/graph-size-dist-" + str(NUM_GRAPHS) + "-" + str(START_NUM_NODES) + "-" + str(END_NUM_NODES) + "-" + str(STEP) + ".csv"
graph_size_dist.to_csv(file_name, index=True)
print("The program has finished running, and the experiment is complete. The data is saved to", file_name + ".")
