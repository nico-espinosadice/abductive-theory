#!/usr/bin/env python
# coding: utf-8

# # Hypothesis Functionality
# 
# ## Goals
# We want a hypothesis that is:
# - Simple. As few nodes as possible.
# - Probable. Should have a high probability of being true
# 
# ## Important Functions
# - Write the Graph
# - Learn the CPT Table
# - ObserveData: Should be done after CPT Table is created but before hypothesis testing
# - findBestExplanation given the observed data and the graph
# 
# ## TO DO:
# - add functionality that handles multi-node hypothesis where individual hyps don't explain everything

# In[23]:


from similarityfunctions import *
import networkx as nx
from itertools import permutations, combinations
import sys, os
from collections import Counter
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



# In[14]:


# add observations to a graph
def observeData(graph, true_nodes, false_nodes):
    nodes = graph.nodes()
    for i in nodes:
        if i in true_nodes:
            graph.nodes[i]['value'] = 1
        elif i in false_nodes:
            graph.nodes[i]['value'] = 0
        """else:
            graph.nodes[i]['value'] = None"""


# In[15]:


class CPT(object):
    """
    Defines a CPT Class
    """
    def __init__(self, num_parents):
        self.num_parents = num_parents
        self.CPTable = self.make_table()
        
    def make_table(self):
        CPTable = {}
        if self.num_parents > 0:
            for i in range(2**self.num_parents):
                CPTable[bin(i)] = 0.0
        else:
            CPTable['self'] = 0.0
        return CPTable
    
    def add_entry(self, parent_values, prob):
        # values of nodes sorted in alphabetical order
        key = ''
        for i in parent_values:
            key += str(i)
        self.CPTable[bin(int(key, 2))] = prob 
    
    def add_entry_self(self, prob):
        self.CPTable['self'] = prob
    
    def get_entry(self, parent_values, value):
        key = ''
        for i in parent_values:
            key += str(i)
        return self.CPTable[bin(int(key, 2))][value]
    
    def get_entry_bin(self, parent_bin, value):
        return self.CPTable[parent_bin][value]
    
    def get_entry_self(self, value):
        return self.CPTable['self'][value]
    
    def get_table(self):
        return self.CPTable


# In[16]:


from itertools import product
def calculateMarginalProbability(blanket, knodes):
    sorted_nodes = sorted(blanket.nodes())
    # want to calculate P(hyp | obs)
    # first, calculate marginal probability of P(hyp, obs, nodes)
    
    unodes = [item for item in sorted_nodes if (not item in knodes)]
    keys = list(product(range(2), repeat=len(unodes)))
    prob = 0
    for i in keys:
        p = 1
        for x in sorted_nodes:
            # find the parent values of a node
            parents = sorted(list(blanket.predecessors(x)))
            # if it has parents, get the key and get the probability
            if parents:
                # get the parents:
                parent_values = []
                for z in parents:
                    if z in knodes:
                        parent_values.append(blanket.nodes[z]['value'])
                    else:
                        parent_values.append(i[unodes.index(z)])
                
                parents_key = bin(int(''.join(map(str, parent_values)), 2))
                if x in knodes:
                    p *= blanket.nodes[x]['CPT'].get_entry_bin(parents_key, blanket.nodes[x]['value'])
                else:
                    p *= blanket.nodes[x]['CPT'].get_entry_bin(parents_key, i[unodes.index(x)])
            # if it doesn't have parents, get the entry for itself based on the node value
            else:
                if x in knodes:
                    p *= blanket.nodes[x]['CPT'].get_entry_self(blanket.nodes[x]['value'])
                else:
                    p *= blanket.nodes[x]['CPT'].get_entry_self(i[unodes.index(x)])
        prob += p
    return prob


# In[17]:


def calculateTotalMarginalProbability(blanket, hyp, obs, bnodes):
    # hyp | obs
    # obs | hyp
    all_obs = []
    all_obs += obs
    all_obs += hyp
    all_obs += bnodes
    new_obs = obs + bnodes
    num_prob = calculateMarginalProbability(blanket, all_obs)
    denom_prob = calculateMarginalProbability(blanket, new_obs)
    return num_prob / denom_prob

def calculateTotalMarginalProbabilityObs(blanket, hyp, obs, bnodes):
    all_obs = []
    all_obs += obs
    all_obs += hyp
    all_obs += bnodes
    new_hyp = hyp + bnodes
    num_prob = calculateMarginalProbability(blanket, all_obs)
    denom_prob = calculateMarginalProbability(blanket, new_hyp)
    return num_prob / denom_prob


# In[140]:


# Finds all possible hypothesis
def combinationsList(my_list):
    total = []
    for i in range(1, len(my_list) + 1):
        total += list(combinations(my_list, i))
    return total

"""def findHypotheses(graph, observed_nodes, bnodes):
    all_node_combos = combinationsList(graph.nodes())
    hyps = []
    for i in all_node_combos:
        children = {}
        for x in i:
            children.update(get_all_children(graph, x, {}, 1))
        if all(item in children.keys() for item in observed_nodes):
            if not any(item in observed_nodes for item in i) and not any(item in bnodes for item in i) :
                hyps.append(list(i))
    return hyps"""

def findHypotheses(graph, observed_nodes, bnodes):
    all_node_combos = combinationsList(graph.nodes())
    hyps = []
    for i in all_node_combos:
        children = {}
        for x in i:
            children.update(get_all_children(graph, x, {}, 1))
        if all(item in children.keys() for item in observed_nodes):
            if not any(item in observed_nodes for item in i) and not any(item in bnodes for item in i) :
                hyps.append(list(i))
                
    # check similarity
    for h in hyps:
        sims = []
        if (len(h) > 1):
            for n in h:
                other_nodes = [x for x in h if (x != n)]
                for o in other_nodes:
                    sim = calculate_similarity(graph, n, o)
                    sims.append(sim)
        too_sim = [i for sub in sims for i in sub if (i > .5)]
        #too_sim = [s for s in sim for simVal in sims if (s > .5)]
        if too_sim:
            hyps.remove(h)
            
    # add best single best hypotheses
    merged_hyps = list(itertools.chain.from_iterable(hyps))
    single_hyps = list(set(merged_hyps))
    hyp_desc = {h:len(list(nx.descendants(graph, h))) for h in single_hyps}
    best_single_hyp = getMaxKeys(hyp_desc)
    hyps += [h for h in best_single_hyp]
    
    return hyps



# Finds a bunch of hypothesis. Does nothing yet
def findBestExplanation(graph, observed_nodes, bnodes, flipped=False):
    hyps = findHypotheses(graph, observed_nodes, bnodes)
    current_best = (None, 0)
    for i in hyps:
        if isinstance(i, list): 
            for x in i:
                graph.nodes[x]['value'] = 1
            if flipped:
                prob = calculateTotalMarginalProbabilityObs(graph, list(i), observed_nodes, bnodes)
            else:
                prob = calculateTotalMarginalProbability(graph, list(i), observed_nodes, bnodes)
            for x in i:
                graph.nodes[x]['value'] = None
        else:
            graph.nodes[i]['value'] = 1
            if flipped:
                prob = calculateTotalMarginalProbabilityObs(graph, list(i), observed_nodes, bnodes)
            else:
                prob = calculateTotalMarginalProbability(graph, list(i), observed_nodes, bnodes)
            graph.nodes[i]['value'] = None
        if prob > current_best[1]:
            current_best = (i, prob)
        print((i, prob))
    return current_best


# ## Cost Functions
# - What do we want to prioritize? 
# 

# ## Testing

# ### Burgler Tests
# Based off the example found [HERE](https://www.ics.uci.edu/~rickl/courses/cs-171/2012-wq-cs171/2012-wq-cs171-lecture-slides/2012wq171-17-BayesianNetworks.pdf)

# In[141]:




# ## Stuff we're probably not using

# In[138]:


# adds new true nodes and new false nodes to the graph and then checks for contradictions
# only finds direct contraidctions? Should talk with group about this.
import copy 
def findContradictions(graph, new_true_nodes, new_false_nodes, threshold=.5):
    new_g = copy.deepcopy(graph)
    observeData(new_g, new_true_nodes, new_false_nodes)
    edge_attrs = nx.get_edge_attributes(new_g, 'weight')
    for i in new_g.nodes():
        if 'observed' in new_g.nodes.data()[i]:
            if new_g.nodes.data()[i]['observed'] is True:
                for x in edge_attrs:
                    print(x, x[1], new_g.nodes.data()[x[1]], edge_attrs[x])
                    if 'observed' in new_g.nodes.data()[x[1]]:
                        if x[0] == i and new_g.nodes.data()[x[1]]['observed'] is False and edge_attrs[x] >= threshold:
                            return (i, True, x[1], False, edge_attrs[x])
    return "No contradictions found"


# In[139]:


# sees if some nodes are independent based on the nodes we are conditioning on
# we are only checking for *direct* conditions. chains are not accounted for
# since in baysien networks edges are "direct dependence." Should double-check this.
def independenceChecker(graph, conditions, nodes):
    edges = nx.get_edge_attributes(graph, 'weight')
    for i in edges:
        if i[1] in nodes and not i[0] in conditions:
            return False
    return True


# In[ ]:


def getMarkovBlanket(graph, obs, hyp):
    #generates a new graph (markov blanket) based on the graph, observations, and hypothesis
    #get all parent nodes
    edge_attrs = nx.get_edge_attributes(graph, 'weight')
    new_graph = nx.DiGraph()
    new_graph.add_node(hyp)
    new_graph.add_nodes_from(obs)
    for i in edge_attrs:
        if i[0] in obs or i[0] == hyp:
            new_graph.add_node(i[1])
            new_graph.add_edge(i[0], i[1], weight=edge_attrs[i])
        if i[1] in obs or i[1] == hyp:
            new_graph.add_node(i[0])
            new_graph.add_edge(i[0], i[1], weight=edge_attrs[i])
    return new_graph



############### HANA
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_all_FBE(graph, observations, bnodes, flipped=False):
    hypotheses = findHypotheses(graph, observations, bnodes)
    not_hyp = bnodes + observations
    possibly_true = []
    for i in range(1, len(not_hyp) + 1):
        possibly_true += [list(x) for x in it.combinations(not_hyp, i)]
    possibly_true

    fbe = {}
    for t in possibly_true:
        false_nodes = list(set(not_hyp) - set(t))
        observeData(graph, t, false_nodes)
        if (flipped == False): fbe[tuple(t)] = findBestExplanation(graph, observations, bnodes)
        elif (flipped==True): fbe[tuple(t)] = findBestExplanation(graph, observations, bnodes, flipped=True)
    
    return fbe


# get all possible combinations of true/false nodes
def get_possible_truths(graph, bnodes, observations):
    not_hyp = bnodes + observations
    
    possibly_true = []
    for i in range(1, len(not_hyp) + 1):
        possibly_true += [list(x) for x in it.combinations(not_hyp, i)]
    
    true_false_nodes = {}
    for t in possibly_true:
        false_nodes = list(set(not_hyp) - set(t))
        true_false_nodes[tuple(t)] = false_nodes
        
    return true_false_nodes


############# Using similarity for edge generation
#accepts a node and list of nodes, returns the most similar node from the list
def most_similar_nodes(graph, node1, list_of_nodes):
    JI, IS, FS = {}, {}, {}
    for n in list_of_nodes: 
        JI[(node1, n)] = calculate_jaccard_similarity(graph, node1, n)
        IS[(node1, n)] = get_immediate_similarity(graph, node1, n)
        FS[(node1, n)] = get_full_similarity(graph, node1, n)
        
    max_JI = max(JI, key=JI.get)
    #max_IS = max(IS, key=IS.get)
    max_FS = max(FS, key=FS.get) 
    
    most_similar = [m[1] for m in [max_JI, max_IS, max_FS]]
    
    return most_similar




# adapted from source: https://thispointer.com/python-how-to-get-all-keys-with-maximum-value-in-a-dictionary/
def getMaxKeys(dictionary):
    # Find item with Max Value in Dictionary
    itemMaxValue = max(dictionary.items(), key=lambda x: x[1])

    listOfKeys = list()
    # Iterate over all the items in dictionary to find keys with max value
    for key, value in dictionary.items():
        if value == itemMaxValue[1]:
            listOfKeys.append(key)
    
    return listOfKeys


def fbeCount(fbe_dict, hypotheses):
    fbe_results = []
    for v in list(fbe_dict.values()):
        if isinstance(v[0], list): fbe_results.append(tuple(v[0]))
        else: fbe_results.append(v[0])
    fbe_count = dict(Counter(fbe_results))
    
    # add 0 count hypotheses
    for h in hypotheses:
        if isinstance(h, list) and tuple(h) not in list(fbe_count.keys()): fbe_count[tuple(h)] = 0
        elif not isinstance(h, list) and h not in list(fbe_count.keys()): fbe_count[h] = 0
    return fbe_count


# get the observed descendants of an hypothesis
def getHypDesc(graph, hypotheses, observations):
    nodes = list(graph.nodes())
    obs_desc_dict = {}
    for h in hypotheses:
        if isinstance(h, list):
            obs_desc = []
            for n in h:
                descendants = list(nx.descendants(graph, n))
                obs_desc += list(set(descendants) & set(observations))
            obs_desc = list(set(obs_desc))
            obs_desc_dict[tuple(h)] = obs_desc
        else: 
            descendants = list(nx.descendants(graph, h))
            obs_desc = list(set(descendants) & set(observations))
            obs_desc_dict[h] = obs_desc
    return obs_desc_dict


def getHypData(graph, hypotheses, observations, bnodes):
    fbe1 = get_all_FBE(graph, observations, bnodes)
    fbe2 = get_all_FBE(graph, observations, bnodes, flipped=True)
    
    # for each hyp, count the number of times it is the best explanation
    count1 = fbeCount(fbe1, hypotheses)
    count2 = fbeCount(fbe2, hypotheses)
    
    # for each hyp, count the number of its observed descendants
    hyp_desc_dict = getHypDesc(graph, hypotheses, observations)
    desc_count = {k:len(v) for k,v in hyp_desc_dict.items()}
    
    # for each hyp, create a list of the following format: [# of obs descendants, count for P(O|H), count for P(H|O)]
    hyp_dict = {}
    for h in hypotheses:
        if not isinstance(h, list): hyp_dict[h] = [desc_count.get(h), count1.get(h), count2.get(h)]
        else: hyp_dict[tuple(h)] = [desc_count.get(tuple(h)), count1.get(tuple(h)), count2.get(tuple(h))]
    
    return hyp_dict

################################## NOT USEFUL


def most_similar_parents(graph, node1, node2):
    JI, IS, FS = {}, {}, {}
    parents1 = list(graph.predecessors(node1))
    parents2 = list(graph.predecessors(node2))
    for p in parents2: 
        JI[(node1, p)] = calculate_jaccard_similarity(graph, node1, node2)
        IS[(node1, p)] = get_immediate_similarity(graph, node1, node2)
        FS[(node1, p)] = get_full_similarity(graph, node1, node2)
        
    max_JI = max(JI, key=JI.get)
    #max_IS = max(IS, key=IS.get)
    max_FS = max(FS, key=FS.get) 
        
    #return max_JI, max_IS, max_FS
    return max_JI, max_FS


# find differences in descendants between best hyp and similar hypotheses
def findDiffDesc(dg, node1, list_of_nodes):
    hyp_nodes = list(set(hyp_nodes) - set(bh))
    bh_desc = list(nx.descendants(dg, bh))
    similar_hyp = similar_to_best.get(bh)
    
    diff_desc = []
    for sh in similar_hyp:
        sim_desc = list(nx.descendants(dg, sh))
        difference = list(set(sim_desc) - set(bh_desc))
        diff_desc += difference
    if diff_desc:
        for o in diff_desc:
            dg.add_edge(bh, o)
            print(nx.is_directed_acyclic_graph(dg))
    else: 
        print('No different descendants')

        
        


