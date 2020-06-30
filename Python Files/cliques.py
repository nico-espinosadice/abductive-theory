#!/usr/bin/env python
# coding: utf-8

#### File: 

# Import necessary libraries
from jaccard_index import *
from edit_distance import *


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