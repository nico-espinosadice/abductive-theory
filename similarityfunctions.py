#!/usr/bin/env python
# coding: utf-8

# # Graphical Models: Similarity Distribution Experimentation

# In[1]:


# Import necessary libraries
import networkx as nx
from networkx.algorithms import approximation
from networkx.algorithms.approximation import independent_set, clique

from matplotlib import *
import matplotlib.pyplot as plt
import random, itertools as it
from scipy.special import comb
import pandas as pd
import numpy as np
import itertools
from sklearn.datasets import make_blobs
import statistics
from statistics import *
import future
from future import *
import sys
import itertools as it
from matplotlib.pyplot import figure
from collections import OrderedDict



# ## Similarity Implementations


###################### GENERAL METHODs ############


def get_all_children(graph, parent, children_dict={}, carry=1):
    carry = 1
    weights = 5
    
    weights = nx.get_edge_attributes(graph, 'weight')
    for i in graph.out_edges(parent):
        children_dict[i[1]] = carry*weights[i]
        get_all_children(graph, i[1], children_dict=children_dict, carry=weights[i])
    return children_dict


# (Adapted from function above)
# Get all children in a dictionary including their respective weights (chains taken into account)
def get_descendants(graph, parent, children_dict={}, carry=1):
    weights = nx.get_edge_attributes(graph, 'weight')
    
    for i in graph.out_edges(parent):
        children_dict[i[1]] = carry*weights[i]
        get_descendants(graph, i[1], children_dict=children_dict, carry=weights[i])
    
    return children_dict


# Returns children of a single node
def get_children(gm, parent):
    children_weights = {}
    weights = nx.get_edge_attributes(gm, 'weight')
    
    for i in gm.out_edges(parent):
        children_weights[i[1]] = weights[i]

    return children_weights



def find_node_children_subgraph(graph, snode, childgraph):
    cnodes = graph.neighbors(snode)
    edges = nx.edges(graph)
    for i in cnodes:
        childgraph.add_node(i)
        childgraph.add_edge(snode, i)
        childgraph = find_node_children_subgraph(graph, i, childgraph)
    return childgraph


######################### JACCARD INDEX #########################

# In[6]:


# all children of rain
def calculate_intersection(graph, children_a, children_b):
    inter = 0
    for i in children_a:
        if i in children_b:
            if children_a[i] < children_b[i]:
                inter += children_a[i]
            else:
                inter += children_b[i]
    return inter  


# In[7]:


# helper func to calculate the union using method 1
def calculate_union_1(graph, children_a, children_b):
    union = 0
    for i in children_a:
        union += children_a[i]
    for i in children_b:
        union += children_b[i]
    union -= calculate_intersection(graph, children_a, children_b)
    return union


# In[8]:


# helper func to calculate the union using method 2
def calculate_union_2(graph, children_a, children_b):
    union = 0
    for i in children_a:
        if i in children_b:
            union += abs(children_a[i] - children_b[i])
        else:
            union += children_a[i]
    for i in children_b:
        if not i in children_a:
            union += children_b[i]
    return union


# In[9]:



# calculate the similarity, compares both methods
def calculate_jaccard_similarity(graph, node_a, node_b):
    children_a = get_all_children(graph, node_a, {})
    children_b = get_all_children(graph, node_b, {})
    intersection = calculate_intersection(graph, children_a, children_b)
    union = calculate_union_1(graph, children_a, children_b)
    if union == 0:
        total = 0
    else:
        total = intersection/union
    return total
    
    


################################ EDIT DISTANCE #################################

# Edit-distance similarity implementation based only on a node's children.

# In[10]:


# Returns the similarity of node B compared to node A, based on 
# children
def get_immediate_similarity(gm, A, B):
    cost = 0.0
    A_children = get_children(gm, A)
    B_children = get_children(gm, B)
    
    if len(A_children) > 0:
        for child in A_children:
            if child in B_children:
                diff = abs(A_children[child] - B_children[child])
                cost += diff
            else:
                cost += 1

        return cost / len(A_children)
    
    else:
        return 0


# In[11]:


# Returns the node that is most similar to the input node
def find_most_immediate_similar(gm, A):
    nodes = list(gm.nodes)
    nodes.remove(A)
    
    max_similarity = get_immediate_similarity(gm, A, nodes[0])
    max_sim_node = nodes[0]
    for node in nodes:
        node_similarity = get_immediate_similarity(gm, A, node)
        
        if node_similarity < max_similarity:
            max_sim_node = node
            max_similarity = node_similarity
    
    return max_sim_node


# Edit-distance similarity implementation based on all descendants of a node.

# In[12]:


# Returns the similarity of node B compared to node A
def get_full_similarity(gm, A, B):
    cost = 0.0
    A_children = get_descendants(gm, A, {}, 1)
    B_children = get_descendants(gm, B, {}, 1)
    
    if len(A_children) > 0:
        for child in A_children:
            if child in B_children:
                diff = abs(A_children[child] - B_children[child])
                cost += diff
            else:
                cost += 1

        return cost / len(A_children)
    
    else:
        return 0

# Optional print line in for loop for debugging: 
# print(child, A_children[child], B_children[child])


# In[13]:


# Returns the node that is most similar to the input node
def find_most_similar(gm, A):
    nodes = list(gm.nodes)
    nodes.remove(A)
    
    max_similarity = get_full_similarity(gm, A, nodes[0])
    max_sim_node = nodes[0]
    for node in nodes:
        node_similarity = get_full_similarity(gm, A, node)
        
        if node_similarity < max_similarity:
            max_sim_node = node
            max_similarity = node_similarity
    
    return max_sim_node


# ### NetworkX: Edit-Distance

# In[14]:


# nx.graph_edit_distance(Graph1, Graph2)


# ### NetworkX: SimRank

# In[15]:


# nx.algorithms.similarity.simrank_similarity(graph, source="node_a", target="node_b")


########################################## GENERATING RANDOM WEIGHTED DAGs ############################3

# In[16]:


def addNodes(gm, num_nodes):
    node_list = []
    
    for i in range(num_nodes):
        gm.add_node(i)

    return gm

# NOTE: This function does not have functionality implemented yet that 
# prevents cycles from occurring
def addEdges(gm, num_nodes, num_edges):
    for i in range(num_edges):
        parent = random.randrange(num_nodes)
        child = random.randrange(num_nodes)
        
        conditional_prob = random.uniform(0, 1)
        gm.add_edge(parent, child, weight=conditional_prob)
            
        if not nx.is_directed_acyclic_graph(gm):
            gm.remove_edge(parent, child)
        
    return gm

# def find_paths(graph, start, end, path=[]):
#     if start == end:
#         return path
    
#     children = get_children(graph, start)
    
#     for node in children:
#         find_paths(graph, start)

def makeDirectedGraph(num_nodes, num_edges):
    dg = nx.DiGraph() # creates directed graph
    dg = addNodes(dg, num_nodes)
    dg = addEdges(dg, num_nodes, num_edges)
    return dg


# In[21]:


# ## Running Experiments

# ### Distribution of Similarity Over Random Graphs of Constant Size

# In[22]:


def getSimilarityData(gm):
    node_list = list(gm.nodes)
    node_pairs = list(it.combinations(node_list, 2))

    ed_immediate_sim, ed_full_sim, ji_sim_1, ji_sim_2 = [], [], [], []

    for pair in node_pairs:
        ed_immediate_sim.append(get_immediate_similarity(gm, pair[0], pair[1]))
        ed_immediate_sim.append(get_immediate_similarity(gm, pair[1], pair[0]))

        ed_full_sim.append(get_full_similarity(gm, pair[0], pair[1]))
        ed_full_sim.append(get_full_similarity(gm, pair[1], pair[0]))

        ji_sim_1.append(calculate_similarity(gm, pair[0], pair[1])["method 1"])
        ji_sim_1.append(calculate_similarity(gm, pair[1], pair[0])["method 1"])

        ji_sim_2.append(calculate_similarity(gm, pair[0], pair[1])["method 2"])
        ji_sim_2.append(calculate_similarity(gm, pair[1], pair[0])["method 2"])

    similarity_data = {"Edit-Distance Immediate Similarity": ed_immediate_sim, "Edit-Distance Full Similarity": ed_full_sim,
           "Jaccard Index Similarity (Method 1)": ji_sim_1, "Jaccard Index Similarity (Method 2)": ji_sim_2}
        
    return similarity_data


# In[23]:


def getCliqueData(gm):
    node_list = list(gm.nodes)
    num_rows = len(list(it.combinations(node_list, 2))) * 2
    
    longest_max_indep_set = len(independent_set.maximum_independent_set(gm))
    max_cliques = list(clique.clique_removal(gm)[1]) * num_rows
    num_max_cliques = len(max_cliques)
    longest_max_clique = len(max_cliques[0])
    
    clique_data = {"Longest Maximum Independent Set": [longest_max_indep_set] * num_rows,
                   "Number of Maximum Cliques": [num_max_cliques] * num_rows, 
                   "Longest Maximum Clique": [longest_max_clique] * num_rows}
    
    return clique_data


# In[24]:


def getSimilarityDistribution(num_graphs, num_nodes, num_edges): 
    graph_df_list = []
    
    for i in range(num_graphs):
        dg = makeDirectedGraph(num_nodes, num_edges)
        
        similarity_data = getSimilarityData(dg)
        clique_data = getCliqueData(dg)
                
        similarity_data.update(clique_data)
                        
        graph_df = pd.DataFrame(data = similarity_data)
        
        graph_df_list.append(graph_df)
    
    return graph_df_list


# In[25]:


# similarity_dist = getSimilarityDistribution(10, 30, 45)
# similarity_dist[0]


# ### Condensing Data

# In[26]:


# condenseExperimentData() takes the output of runExperiment as an input.
# Returns a DataFrame consisting of averages for each similarity index
def condenseExperimentData(experiment_data):
    ed_immediate_sim_avgs, ed_full_sim_avgs, ji_sim_1_avgs, ji_sim_2_avgs = [], [], [], []
    longest_max_ind_set_avgs, num_max_cliques_avgs, longest_max_clique_avgs = [], [], []
    
    for graph_df in experiment_data:
        graph_avgs = graph_df.mean(axis=0)
        ed_immediate_sim_avgs.append(graph_avgs["Edit-Distance Immediate Similarity"])
        ed_full_sim_avgs.append(graph_avgs["Edit-Distance Full Similarity"])
        ji_sim_1_avgs.append(graph_avgs["Jaccard Index Similarity (Method 1)"])
        ji_sim_2_avgs.append(graph_avgs["Jaccard Index Similarity (Method 2)"])
        longest_max_ind_set_avgs.append(graph_avgs["Longest Maximum Independent Set"])
        num_max_cliques_avgs.append(graph_avgs["Number of Maximum Cliques"])
        longest_max_clique_avgs.append(graph_avgs["Longest Maximum Clique"])
        
    condensed_data = {"Edit-Distance Immediate Similarity Averages": ed_immediate_sim_avgs,
                     "Edit-Distance Full Similarity Averages": ed_full_sim_avgs,
                     "Jaccard Index Similarity (Method 1) Averages": ji_sim_1_avgs,
                     "Jaccard Index Similarity (Method 2) Averages": ji_sim_2_avgs,
                     "Longest Maximum Independent Set": longest_max_ind_set_avgs,
                     "Number of Maximum Cliques": num_max_cliques_avgs,
                     "Longest Maximum Clique": longest_max_clique_avgs}
    condensed_df = pd.DataFrame(data = condensed_data)
    
    return condensed_df


# In[27]:


# condensed_similarity_dist = condenseExperimentData(similarity_dist)
# condensed_similarity_dist


# ### Distribution Over Varying Graph Sizes

# In[28]:


nodes_to_edges_factor = 1.5
def getGraphSizeDistribution(num_graphs, start_num_nodes, end_num_nodes, step):
    experiments_df = pd.DataFrame(columns = ["Nodes", "Edit-Distance Immediate Similarity",
                                            "Edit-Distance Full Similarity",
                                            "Jaccard Index Similarity (Method 1)",
                                            "Jaccard Index Similarity (Method 2)",
                                            "Longest Maximum Independent Set",
                                            "Number of Maximum Cliques",
                                            "Longest Maximum Clique"])
    
    for num_nodes in range(start_num_nodes, end_num_nodes, step):
        num_edges = int(num_nodes * 1.5)
        experiment_data = getSimilarityDistribution(num_graphs, num_nodes, num_edges)
        condensed_data = condenseExperimentData(experiment_data)
        condensed_data_avgs = condensed_data.mean(axis=0)

        experiments_df = experiments_df.append({"Nodes": num_nodes,
                               "Edit-Distance Immediate Similarity": condensed_data_avgs["Edit-Distance Immediate Similarity Averages"],
                                "Edit-Distance Full Similarity": condensed_data_avgs["Edit-Distance Full Similarity Averages"],
                                "Jaccard Index Similarity (Method 1)": condensed_data_avgs["Jaccard Index Similarity (Method 1) Averages"],
                                "Jaccard Index Similarity (Method 2)": condensed_data_avgs["Jaccard Index Similarity (Method 2) Averages"],
                                "Longest Maximum Independent Set": condensed_data_avgs["Longest Maximum Independent Set"],
                                "Number of Maximum Cliques": condensed_data_avgs["Number of Maximum Cliques"],
                                "Longest Maximum Clique": condensed_data_avgs["Longest Maximum Clique"]}, ignore_index=True)

    return experiments_df


################### Reversed Graphs

def reverseEdgeWeight(conditional_prob):
    '''Accepts a conditional probability value P(A|B), returns P(B|A) and a pair of values for P(A) and P(B) that
        would be consistent with the given conditional probability and its reversed conditional probability.)'''
    prior = random.uniform(0,1)
    marginal_prob = random.uniform(0,1)
    
    likelihood = (conditional_prob*marginal_prob)/prior
    known_conditional_prob = (prior*likelihood)/marginal_prob
    
    if (known_conditional_prob != conditional_prob):
        reverseEdgeWeight(conditional_prob)
    else: return likelihood
        


# reverses the order of a tuple
def Reverse(tuples): 
    new_tup = tuples[::-1] 
    return new_tup 

# returns the reverse of a weighted graph
def reverseWeightedDG(dg):
    rdg = nx.reverse_view(dg)
    
    for e in rdg.edges():
        dg_edge = Reverse(e)
        dg_weight = dg[e[1]][e[0]]['weight']
        
        rdg_weight = reverseEdgeWeight(dg_weight)
        rdg[e[0]][e[1]]['weight'] = rdg_weight
        
    return rdg




########### Edit Distance
# accepts DAG and its reversed graph, returns a dictionary {ed (imm sim) of the DAG : ed (imm sim) of its reverse}
def dg_rdg_immSim(dg, rdg):
    dg_nodes = nx.nodes(dg)
    dg_weights = nx.get_edge_attributes(dg, 'weight')
    dg_nodePairs = list(it.combinations(dg_nodes, 2))

    dg_rdg_data = {}
    dg_is = []
    rdg_is = []
    for pair in dg_nodePairs:
        dg_rdg_data[get_immediate_similarity(dg, pair[0], pair[1])] = get_immediate_similarity(rdg, pair[0], pair[1])
        dg_rdg_data[get_immediate_similarity(dg, pair[1], pair[0])] = get_immediate_similarity(rdg, pair[1], pair[0])
        
    return dg_rdg_data
        
    
# accepts DAG and its reversed graph, returns a dictionary {ed (full sim) of the DAG : ed (full sim) of its reverse}
def dg_rdg_fullSim(dg, rdg):
    dg_nodes = nx.nodes(dg)
    dg_weights = nx.get_edge_attributes(dg, 'weight')
    dg_nodePairs = list(it.combinations(dg_nodes, 2))

    dg_rdg_data = {}
    dg_is = []
    rdg_is = []
    for pair in dg_nodePairs:
        dg_rdg_data[get_full_similarity(dg, pair[0], pair[1])] = get_full_similarity(rdg, pair[0], pair[1])
        dg_rdg_data[get_full_similarity(dg, pair[1], pair[0])] = get_full_similarity(rdg, pair[1], pair[0])
        
    return dg_rdg_data


################ Jaccard Index

#get list of Jaccard Indices for each node pair in a graph
def jaccardDG(dg):
    nodes = list(dg.nodes)   #list of dag nodes
    node_pairs = list(it.combinations(nodes, 2))

    jindex = []
    for pair in node_pairs:
        jindex.append(calculate_similarity(dg, pair[0], pair[1])["method 1"])
        jindex.append(calculate_similarity(dg, pair[1], pair[0])["method 1"])

    nodes = list(dg.nodes)   #list of dag nodes
    node_pairs = list(it.combinations(nodes, 2))
    return jindex


# get avg Jaccard Index for all node pairs in a graph
def avgJaccard(dg):
    nodes = list(dg.nodes)   #list of dag nodes
    node_pairs = list(it.combinations(nodes, 2))

    jindex = []
    for pair in node_pairs:
        jindex.append(calculate_similarity(dg, pair[0], pair[1])["method 1"])
        jindex.append(calculate_similarity(dg, pair[1], pair[0])["method 1"])

    nodes = list(dg.nodes)   #list of dag nodes
    node_pairs = list(it.combinations(nodes, 2))
    return mean(jindex)



######################## CLIQUES ############################
#Input desired number of cliques, output DAG with that many cliques
def dgWithCliqs(num_cliq):   
    #create DAG
    rand_num_nodes = random.randint(num_cliq, num_cliq*3)
    dg = makeDirectedGraph(rand_num_nodes, int(rand_num_nodes*1.5))
    
    #count num of cliques in graph
    cliq_rem = clique.clique_removal(dg)
    cliqCount = len(cliq_rem[1])   
    
    # remake graph if it doesn't match the input
    if (cliqCount == num_cliq):
        return dg
    else:
        return dgWithCliqs(num_cliq)



#function that accepts a graph, returns a list of all the immediate similarity values between clique pairs. 
#Do the same for node pairs.

def immSimCliqs(dg):
    cliq_rem = clique.clique_removal(dg)
    maxCliqs = cliq_rem[1]
    cliqueChoose2 = list(itertools.combinations(maxCliqs, 2))
    cliq_imm_sim = []
    for pair in cliqueChoose2:
        imm_sim = get_immediate_similarity(dg, pair[0], pair[1])
        cliq_imm_sim.append(imm_sim)
        #print(pair, imm_sim)
    return cliq_imm_sim
    
    
    
