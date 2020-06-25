#!/usr/bin/env python
# coding: utf-8

#### Similarity Distribution Over Random Graphs of Varying Size

# Import necessary libraries
from exp-sim-dist import *


### Calculating Similarity Distribution Over Graph Sizes

def getGraphSizeDistribution(num_graphs, start_num_nodes, end_num_nodes, step):    
    cols = pd.MultiIndex.from_product([column_list, metrics])
    graph_size_dist = pd.DataFrame(index = [["Number of Nodes Per Graph"], ["Number of Graphs"]], columns = cols)

    for num_nodes in range(start_num_nodes, end_num_nodes, step):
        num_edges = int(num_nodes * nodes_to_edges_factor)
        
        graph_similarity_list = getGraphSimilarityList(num_graphs, num_nodes, num_edges)
        graph_distribution = getCondensedSimilarityDistribution(graph_similarity_list)
        
        new_row = getNewRow(graph_distribution)
        graph_size_dist.loc[(num_nodes, num_graphs),:] = new_row

    return graph_size_dist


## Helper Functions
def getNewRow(graph_distribution): 
    new_row = []
    
    for column in column_list:
        similarities = graph_distribution[column]["mean"]
        new_row.append(similarities.mean())
        new_row.append(similarities.median())
        new_row.append(similarities.std())

    return tuple(new_row)



### Exporting Data
# To do



### Running File
# To do