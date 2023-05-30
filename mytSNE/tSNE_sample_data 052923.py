#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import mytSNE_052923


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

colors = np.array(["orange", "orange","orange","orange", "blue", "blue","blue","blue","red","red","red","red",])
plt.scatter(data[:,0], data[:,1], c=colors)
plt.show()


# In[3]:


mytSNE_052923.runTSNE(data, colors, 3, 30)
print()


# In[5]:


#generate data and run
num_clusters = 4
points_per_cluster = 4
dimensionality = 5

np.random.seed(1)

# Generate random cluster centroids
centroids = 10*np.random.randn(num_clusters, dimensionality)

randArr = np.random.randn(points_per_cluster, dimensionality)

# print(centroids[0] + randArr)
    

#make data around each center
data = []
for centroid in centroids:
    cluster_points = centroid + 3*np.random.randn(points_per_cluster, dimensionality)
    
    data.extend(cluster_points)

# Convert the data list to a NumPy array
data_matrix = np.array(data)

print(data_matrix)
print()
print()
colors = ["red"]*4 + ["blue"]*4 + ["green"]*4 + ["orange"]*4


# In[7]:


lowDimData = mytSNE_052923.runTSNE(data_matrix, colors, perplexity=3, iterations=75, numPCs = 0)
print()


# In[ ]:




