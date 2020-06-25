# ## Analysis

# ### Distribution Over Varying Graph Sizes

# In[42]:


graph_dist = pd.read_csv("Data/graph-size-distrubtion-5-10-41-10.csv")
graph_dist.head()


# In[43]:


fig = plt.figure()
ax = fig.add_subplot(111)

# Adapted from https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

ed_imm_sim = graph_dist["Edit-Distance Immediate Similarity"]
ed_f_sim = graph_dist["Edit-Distance Full Similarity"]
ji_sim = graph_dist["Jaccard Index Similarity (Method 1)"]

ed_imm_sim_std = graph_dist["Edit-Distance Immediate Similarity STD"]
ed_f_sim_std = graph_dist["Edit-Distance Full Similarity STD"]
ji_sim_std = graph_dist["Jaccard Index Similarity (Method 1) STD"]

rects1 = ax.bar(ind, 1 - ed_imm_sim, width, color='r', yerr=ed_imm_sim_std)
rects2 = ax.bar(ind + width, 1 - ed_f_sim, width, color='g', yerr=ed_f_sim_std)
rects3 = ax.bar(ind + width * 2, ji_sim, width, color='b', yerr=ji_sim_std)

ax.set_ylabel('Similarity')
ax.set_xlabel("Graph Size (Number of Nodes)")
ax.set_xticks(ind + width)
ax.set_xticklabels(graph_dist["Nodes"])
ax.legend((rects1[0], rects2[0], rects3[0]), ('Edit-Distance Immediate Similarity', 'Edit-Distance Full Similarity', 'Jaccard Index Similarity (Method 1)'))
ax.set_ylim(0, 1)
ax.set_title("Similarity vs. Graph Size")

plt.show()


# #### Other Work

# In[44]:


graph_size_dist = pd.read_csv("Data/graph-size-distrubtion-5-10-41-10.csv")


# In[45]:


graph_size_dist.head()


# In[46]:


plt.scatter(graph_size_dist["Nodes"], 1 - graph_size_dist["Edit-Distance Immediate Similarity"], label = "Edit-Distance Immediate Similarity")
plt.scatter(graph_size_dist["Nodes"], 1 - graph_size_dist["Edit-Distance Full Similarity"], label = "Edit-Distance Full Similarity")
plt.scatter(graph_size_dist["Nodes"], graph_size_dist["Jaccard Index Similarity (Method 1)"], label = "Jaccard Index Similarity (Method 1)")

plt.xlabel('Size of Graph (Number of Nodes)')
plt.ylabel('Similarity')
plt.title('Similarity vs. Size of Graph')
plt.legend()
plt.rcParams["figure.figsize"] = (10,7)
plt.ylim(0, 1)

plt.show()


# In[47]:


plt.plot(graph_size_dist["Nodes"], graph_size_dist["Longest Maximum Independent Set"], label = "Longest Maximum Independent Set")

plt.xlabel('Size of Graph (Number of Nodes)')
plt.ylabel('Longest Maximum Independent Set')
plt.title('Longest Maximum Independent Set vs. Size of Graph')
plt.legend()

plt.show()


# In[48]:


plt.plot(graph_size_dist["Nodes"], graph_size_dist["Number of Maximum Cliques"], label = "Number of Maximum Cliques")

plt.xlabel('Size of Graph (Number of Nodes)')
plt.ylabel('Number of Maximum Cliques')
plt.title('Number of Maximum Cliques vs. Size of Graph')
plt.legend()

plt.show()


# In[49]:


plt.plot(graph_size_dist["Nodes"], graph_size_dist["Longest Maximum Clique"], label = "Longest Maximum Clique")

plt.xlabel('Size of Graph (Number of Nodes)')
plt.ylabel('Longest Maximum Clique')
plt.title('Longest Maximum Clique vs. Size of Graph')
plt.legend()

plt.show()


# In[50]:


plt.plot(graph_size_dist["Longest Maximum Independent Set"], graph_size_dist["Edit-Distance Full Similarity"], label = "Edit-Distance Full Similarity")

plt.xlabel('Longest Maximum Independent Set')
plt.ylabel('Edit-Distance Full Similarity')
plt.title('Edit-Distance Full Similarity vs. Longest Maximum Independent Set')
plt.legend()
plt.ylim(0, 1)

plt.show()


# In[51]:


plt.plot(graph_size_dist["Longest Maximum Independent Set"], graph_size_dist["Jaccard Index Similarity (Method 1)"], label = "Jaccard Index Similarity (Method 1)")

plt.xlabel('Longest Maximum Independent Set')
plt.ylabel('Jaccard Index Similarity (Method 1)')
plt.title('Jaccard Index Similarity (Method 1) vs. Longest Maximum Independent Set')
plt.legend()
plt.ylim(0, 1)

plt.show()


# ### Distribution of Similarity Over Random Graphs of Constant Size

# In[52]:


sim_dist = pd.read_csv("Data/sim-dist-10-30-45.csv")
sim_dist = sim_dist.rename(columns={"Edit-Distance Immediate Similarity Averages": "Edit-Distance Immediate Similarity", 
                 "Edit-Distance Full Similarity Averages": "Edit-Distance Full Similarity",
                "Jaccard Index Similarity (Method 1) Averages": "Jaccard Index Similarity (Method 1)"})


# In[53]:


sim_dist.head()


# In[54]:


sim_dist["Edit-Distance Immediate Similarity"] = 1 - sim_dist["Edit-Distance Immediate Similarity"]
sim_dist["Edit-Distance Full Similarity"] = 1 - sim_dist["Edit-Distance Full Similarity"]


# In[55]:


sim_dist.boxplot(column=["Edit-Distance Immediate Similarity", "Edit-Distance Full Similarity", 'Jaccard Index Similarity (Method 1)'], figsize = (15,6), return_type = "axes")

plt.ylabel("Similarity")
plt.xlabel("(Using Randomly Generated Directed Graphical Models with 30 nodes and 45 edges)")
plt.title("Similarity of Randomly Generated Directed Graphical Models")
plt.ylim(0, 1)