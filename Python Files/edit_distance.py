#!/usr/bin/env python
# coding: utf-8

#### Similarity: Edit-Distance Implementation

# Import necessary libraries
from graph_generation import *


### Edit-Distance Similarity Implementation: Using only on nodes' immediate children.

# Returns the similarity of node B compared to node A, based on children
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



### Edit-Distance Similarity Implementation: Using all descendants of nodes.

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
        return 1-(cost / len(A_children))
    else:
        return 1


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
