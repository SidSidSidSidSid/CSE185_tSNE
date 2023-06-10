#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import mytSNE_060823 as mytSNE


# In[2]:


data = np.array([[1, 1],
                 [2, 2],
                 [1.5, 1.5],
                 [1.5, 2],
               [1.5, 8],
                 [2.5, 7],
                 [1.5, 7.5],
                 [2, 7.25],
               [6, 7],
                 [7, 6],
                 [6.5, 6.5],
                 [6.7, 6.25]])

colors = np.array(["forestgreen", "limegreen","darkgreen","lime", "navy", "slateblue","lavender","royalblue","maroon","tomato","salmon","lightcoral",])
plt.scatter(data[:,0], data[:,1], c=colors)
plt.show()


# In[3]:


#run TSNE on sample data of 3 clusters of 4 points. Perplexity is 8 and number of iterations is 30

mytSNE.runTSNE(data, 8, 30, numPCs = 0, colors = colors)
print()


# In[4]:


num_clusters = 4
points_per_cluster = 4
dimensionality = 5

# Generate random cluster centroids
centroids = 10*np.random.randn(num_clusters, dimensionality)

randArr = np.random.randn(points_per_cluster, dimensionality)
 

#make data around each center
data = []
for centroid in centroids:
    cluster_points = centroid + 3*np.random.randn(points_per_cluster, dimensionality)
    
    data.extend(cluster_points)

# Convert the data list to a NumPy array
data_matrix = np.array(data)

print(data_matrix)


# In[5]:


#run TSNE on randomely generated sample data of 4 clusters of 4 points. Perplexity is 3 and number of iterations is 200

colors = ["red"]*4 + ["blue"]*4 + ["green"]*4 + ["orange"]*4
mytSNE.runTSNE(data_matrix, 4, 200, numPCs = 0, colors = colors)


# In[ ]:





# In[ ]:




