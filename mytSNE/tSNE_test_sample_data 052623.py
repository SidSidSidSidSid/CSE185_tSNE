#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import mytSNE_052623


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


# In[4]:


mytSNE_052623.runTSNE(data, colors, 8, 200)
print()


# In[5]:


# Set the number of clusters and the number of points per cluster
num_clusters = 4
points_per_cluster = 4

# Set the dimensionality of each data point
dimensionality = 5

# Set the random seed for reproducibility
np.random.seed(1)

# Generate random cluster centroids
centroids = 10*np.random.randn(num_clusters, dimensionality)
print(centroids)

print()
randArr = np.random.randn(points_per_cluster, dimensionality)
print(randArr)

print(centroids[0] + randArr)
    

# Generate data points around each centroid
data = []
for centroid in centroids:
    cluster_points = centroid + np.random.randn(points_per_cluster, dimensionality)
    
    data.extend(cluster_points)

# Convert the data list to a NumPy array
data_matrix = np.array(data)

print(data_matrix)


# In[7]:


colors = ["red"]*4 + ["blue"]*4 + ["green"]*4 + ["orange"]*4
mytSNE_052623.runTSNE(data_matrix, colors, 15, 200)
print()


# In[ ]:




