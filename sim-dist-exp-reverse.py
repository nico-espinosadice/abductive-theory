#!/usr/bin/env python
# coding: utf-8

# In[128]:


from sde import *
import future
from future import *


# In[129]:


# given an edge weight, returns a possible reversed edge weight
def reverseEdgeWeight(conditional_prob):
    '''Accepts a conditional probability value P(A|B), returns P(B|A) and a pair of values for P(A) and P(B) that
        would be consistent with the given conditional probability and its reversed conditional probability.)'''
    prior = random.uniform(0,1)
    marginal_prob = random.uniform(0,1)
    
    likelihood = (conditional_prob*marginal_prob)/prior
    known_conditional_prob = (prior*likelihood)/marginal_prob
    
    if (known_conditional_prob != conditional_prob):
        reverseEdgeWeight(conditional_prob)
    elif (type(likelihood) == type(None)):
        reverseEdgeWeight(conditional_prob)
    else: return likelihood
        


# In[130]:


# returns the reverse of a weighted graph
def reverseWeightedDG(dg):
    rdg = nx.reverse_view(dg)
    
    for e in dg.edges():
        rdg[e[::-1][0]][e[::-1][1]]['weight'] = reverseEdgeWeight(dg[e[0]][e[1]]['weight'])
    
    #print(nx.get_edge_attributes(dg, 'weight'))
    #print(nx.get_edge_attributes(rdg,'weight'))
    return rdg


# In[131]:


# takes in a DAG and returns the immediate edit distance values for both the DAg and its reverse graph
def immediateEDandReverse(dg):
    nodes = list(dg.nodes)   #list of dag nodes
    node_pairs = list(it.combinations(nodes, 2))
    rdg_nodes = list(rdg.nodes)
    rdg_node_pairs = list(it.combinations(rdg, 2))
    
    ed_immediate_sim = []
    for pair in node_pairs:
        ed_immediate_sim.append(get_immediate_similarity(dg, pair[0], pair[1]))
        ed_immediate_sim.append(get_immediate_similarity(dg, pair[1], pair[0]))

    return ed_immediate_sim


# In[132]:


# takes in a DAG and returns the full edit distance values for both the DAg and its reverse graph
def fullEDandReverse(dg, rdg):
    rdg = everseWeightedDG(dg)   #reverse input graph
    nodes = list(dg.nodes)  #node list of OG graph
    node_pairs = list(it.combinations(nodes, 2))   #make all possible node pairs in graph
    
    #get all edit distance similarity values based on node children
    ed_full_sim = []
    for pair in node_pairs:
        ed_full_sim.append(get_full_similarity(dg, pair[0], pair[1]))
        ed_full_sim.append(get_full_similarity(dg, pair[1], pair[0]))
    
    return ed_full_sim


# # Tests

# for graphs with 10, 20, 50, 100, 200, 500, 1000 nodes
# 1.5 times as many edges as nodes

# In[133]:


dg = makeDirectedGraph(10, 15)
rdg = reverseWeightedDG(dg)


# In[134]:


weights = nx.get_edge_attributes(dg, 'weight')
print(weights)


# In[135]:


print(weights.values())


# In[136]:


dict_df = pd.DataFrame({ key:pd.Series(value) for key, value in data.items() })
dict_df


# In[137]:


reversed_dict_df =  pd.DataFrame({ key:pd.Series(value) for key, value in rdata.items() })
reversed_dict_df


# In[143]:


joint_df = dict_df.join(reversed_dict_df)
joint_df


# In[144]:


joint_df.to_excel(r'reverseddg_tests_file.xlsx', index = False, header=True)


# # Jaccard index comparisons

# In[66]:


import statistics
from statistics import *


# In[92]:


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
    


# In[93]:


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
    


# In[120]:


avg_jindex = {}
reverse_avg_jindex = {}

data = {}
rdata = {}


# In[127]:


#get Jaccard indices and averages for graphs with a certain amount of nodes
node_nums = [10,20,50, 100, 500, 1000]
for i in node_nums:
    dg = makeDirectedGraph(i, int(i*1.5))  #make directed graph
    N = str(i)   #convert node amount to string
    #calc and assign avg J Index to data dict
    jAvg = avgJaccard(dg)
    avg_jindex[N + ' node graph average'] = jAvg
    data[N + ' node graph'] = jaccardDG(dg)   #add all Jaccard index data for that size graph
    
    # Repeat for reversed graph
    rdg = reverseWeightedDG(dg)
    #calc and assign avg J Index to data dict
    reversed_jAvg = avgJaccard(dg)
    reverse_avg_jindex[N + ' node graph average'] = reversed_jAvg
    rdata[N + ' node reversed graph'] = jaccardDG(rdg)   #add all Jaccard index data for that size graph  


# In[ ]:


#data.get('10 node graph')


# In[ ]:


df = pd.DataFrame.from_dict(data, orient='index')
df.transpose()


# In[ ]:


dict_df = pd.DataFrame({ key:pd.Series(value) for key, value in data.items() })
dict_df


# In[ ]:


reversed_dict_df =  pd.DataFrame({ key:pd.Series(value) for key, value in rdata.items() })
reversed_dict_df


# # Edit Distance

# In[ ]:


immediateEDandReverse(dg)

