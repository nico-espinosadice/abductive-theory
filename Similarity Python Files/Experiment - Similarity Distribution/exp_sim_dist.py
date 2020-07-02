#!/usr/bin/env python
# coding: utf-8

#### Similarity Distribution Over Random Graphs of Constant Size

# Import necessary libraries
import sys
sys.path.append('../')
from similarity_distribution import *


### Running The Script
## Get Parameters and Constants
parameters = pd.read_csv("parameters-sim-dist.csv")

COLUMN_LIST = parameters["Column List"].to_list()
METRICS = [metric for metric in parameters["Metrics"].to_list() if str(metric) != 'nan']

NODES_TO_EDGES_FACTOR = parameters["Nodes To Edges Factor"][0]
NUM_GRAPHS = int(parameters["Number of Graphs"][0])
NUM_NODES = int(parameters["Number of Nodes"][0])
NUM_EDGES = int(NUM_NODES * NODES_TO_EDGES_FACTOR)


## Run The Experiment
print("Beginning the experiment...")
print("Making", str(NUM_GRAPHS), "with", str(NUM_NODES), "and", str(NUM_EDGESU), "each.")
similarity_list = getGraphSimilarityList(NUM_GRAPHS, NUM_NODES, NUM_EDGES)
similarity_dist = getCondensedSimilarityDistribution(similarity_list, COLUMN_LIST, METRICS)


## Exporting Data
file_name = "Data/sim-dist-" + str(NUM_GRAPHS) + "-" + str(NUM_NODES) + "-" + str(NUM_EDGES) + ".csv"
similarity_dist.to_csv(file_name, index=True)
print("The program has finished running, and the experiment is complete. The data is saved to", file_name + ".")