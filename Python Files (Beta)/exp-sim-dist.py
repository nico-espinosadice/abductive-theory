#!/usr/bin/env python
# coding: utf-8

#### Similarity Distribution Over Random Graphs of Constant Size

### Parameters
column_list = ["Edit-Distance Immediate Similarity", "Edit-Distance Full Similarity", 
                 "Jaccard Index Similarity",
                 "Longest Maximum Independent Set",
                 "Number of Maximum Cliques",
                 "Longest Maximum Clique"]

metrics = ["mean", "50%", "std"]



### Getting Similarity List

def getGraphSimilarityList(num_graphs, num_nodes, num_edges): 
    graph_sim_list = []
    
    for i in range(num_graphs):
        dg = makeDirectedGraph(num_nodes, num_edges)
        
        similarity_data = getSimilarityData(dg)
        clique_data = getCliqueData(dg) 
        similarity_data.update(clique_data)
                        
        graph_sim = pd.DataFrame(data = similarity_data)
        graph_sim_list.append(graph_sim)
    
    return graph_sim_list


## Helper Functions

def getSimilarityData(gm):
    node_list = list(gm.nodes)
    node_pairs = list(it.combinations(node_list, 2))

    ed_imm_sim, ed_full_sim, ji_sim = [], [], []

    for pair in node_pairs:
        ed_imm_sim.append(get_immediate_similarity(gm, pair[0], pair[1]))
        ed_imm_sim.append(get_immediate_similarity(gm, pair[1], pair[0]))

        ed_full_sim.append(get_full_similarity(gm, pair[0], pair[1]))
        ed_full_sim.append(get_full_similarity(gm, pair[1], pair[0]))

        ji_sim.append(calculate_similarity(gm, pair[0], pair[1]))
        ji_sim.append(calculate_similarity(gm, pair[1], pair[0]))

    similarity_data = {"Edit-Distance Immediate Similarity": ed_imm_sim, 
                       "Edit-Distance Full Similarity": ed_full_sim,
                       "Jaccard Index Similarity": ji_sim}
        
    return similarity_data


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



### Creating Similarity Distribution

def getGraphSummaryRow(graph_summary): 
    graph_summary_row = []
    
    for column in column_list:
        graph_summary_row.append(graph_summary[column]["mean"])
        graph_summary_row.append(graph_summary[column]["50%"])
        graph_summary_row.append(graph_summary[column]["std"])
    
    return tuple(graph_summary_row)


def getCondensedSimilarityDistribution(graph_dist_list):
    cols = pd.MultiIndex.from_product([column_list, metrics])
    similarity_dist = pd.DataFrame(index = ["Graph Number"], columns = cols)
    
    index = 0
    for graph_similarity in graph_dist_list:
        graph_summary = graph_similarity.describe(include="all")
        graph_summary_row = getGraphSummaryRow(graph_summary)
    
        similarity_dist.loc[index,:] = graph_summary_row
        index += 1
    
    return similarity_dist



### Exporting Data
# To do