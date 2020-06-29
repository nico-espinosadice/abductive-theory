#!/usr/bin/env python
# coding: utf-8

#### File:

# Import necessary libraries
from jaccard_index import *
from edit_distance import *


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