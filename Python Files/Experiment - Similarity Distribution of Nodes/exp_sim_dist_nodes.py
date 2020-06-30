#!/usr/bin/env python
# coding: utf-8

#### Similarity Distribution Of Nodes (In A Graph)

# Import necessary libraries
import sys
sys.path.append('../')
from similarity_distribution import *

### Running The Script
## Get Parameters and Constants
parameters = pd.read_csv("parameters-sim-dist-nodes.csv")

COLUMN_LIST = parameters["Column List"].to_list()
METRICS = [metric for metric in parameters["Metrics"].to_list() if str(metric) != 'nan']

NODES_TO_EDGES_FACTOR = parameters["Nodes To Edges Factor"][0]
NUM_NODES = int(parameters["Number of Nodes"][0])
NUM_EDGES = int(NUM_NODES * NODES_TO_EDGES_FACTOR)


## Run The Experiment
print("Beginning the experiment...")
print("Making a graph with", str(NUM_NODES), "and", str(NUM_EDGES), "edges.")
sim_dist_nodes = getNodeSimilarityDistribution(NUM_NODES, NUM_EDGES)


## Exporting Data
file_name = "Data/sim-dist-nodes-" + str(NUM_NODES) + "-" + str(NUM_EDGES) + "2.csv"
sim_dist_nodes.to_csv(file_name, index=True)
print("The program has finished running, and the experiment is complete. The data is saved to", file_name + ".")