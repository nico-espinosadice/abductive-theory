#!/usr/bin/env python
# coding: utf-8

#### Similarity: Jaccard Index Implementation

# Import necessary libraries
from graph-generation.py import *


### Calculating Similarity

# calculate the similarity, compares both methods
def calculate_similarity(graph, node_a, node_b):
    children_a = get_all_children(graph, node_a, {})
    children_b = get_all_children(graph, node_b, {})
    
    intersection = calculate_intersection(graph, children_a, children_b)
    union = calculate_union(graph, children_a, children_b)
    
    results = 0
    if union != 0:
        results = intersection/union
            
    return results



### Helper Functions

# Get all children
def calculate_intersection(graph, children_a, children_b):
    inter = 0
    for i in children_a:
        if i in children_b:
            if children_a[i] < children_b[i]:
                inter += children_a[i]
            else:
                inter += children_b[i]
    return inter  


# helper func to calculate the union
def calculate_union(graph, children_a, children_b):
    union = 0
    
    for i in children_a:
        union += children_a[i]
        
    for i in children_b:
        union += children_b[i]
    
    union -= calculate_intersection(graph, children_a, children_b)
    
    return union
