#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
def twodto1dtSNE(numpyarr, colors):
    numData = numpyarr.shape[0]
    
    dataLocs = [0]*numData
    for i in range(len(dataLocs)):
        dataLocs[i] = np.random.rand()
        
    print("On Number Line Random")
    
    plt.scatter(dataLocs, [0]*numData, c=colors)
    plt.show()
    return(dataLocs)

def twodto2dtSNE(numData, colors):    
    dataLocs = np.zeros((numData,2))
    for i in range(len(dataLocs)):
        for j in range(2):
            dataLocs[i,j] = np.random.rand()
        
    print("Random")
    
    plt.scatter(dataLocs[:,0], dataLocs[:,1], c=colors)
    plt.show()
    return(dataLocs)

# data = np.array([[1, 1],
#                  [2, 2],
#                  [1.5, 1.5],
#                  [1.5, 2],
#                [16, 7],
#                  [17, 6],
#                  [16.5, 6.5],
#                  [16.7, 6.25]])
# colors = np.array(["orange", "orange","orange","orange", "blue", "blue", "blue", "blue"])

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

data 

colors = np.array(["orange", "orange","orange","orange", "blue", "blue","blue","blue","red","red","red","red",])




plt.scatter(data[:,0], data[:,1], c=colors)
plt.show()

twodto2dtSNE(data.shape[0], colors)


# In[2]:


def computePairwiseDistance(list1, list2):
    #computes pairwise euclidian distance between 2 lists (must be same length)
    if(len(list1)!=len(list2)):
        print("ERROR! lists must be same length")
        return None
    
    sumOfSquares = 0
    for i in range(len(list1)):
        sumOfSquares += (list1[i]-list2[i])**2
    return math.sqrt(sumOfSquares)
    
    

def computePairwiseDistances(data):
    #takes in a np 2d array where rows are observations and columns are vals for observations and returns pairwise matrix
    #row i, column j is the euclidian distance (had sqrt applied) of the ith point to the jth point
    numSamples = data.shape[0]
    toReturn = np.zeros((numSamples,numSamples))
    toReturn = toReturn - 1
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = computePairwiseDistance(data[i,:], data[j,:])
    return toReturn;

#def normalizeConditionalProbabilities()
    #input a pairwise distance matrix where (i,j) = (row, column) is the euclidian distance of the ith point to the jth point
    #outputs a matrix where (i,j) is P(j|i) that the ith point would pick the jth point as its neighbor if neighbors were picked
    #under a Gaussian distribution centered at xi
    


# In[3]:


def computePairwiseAffinities(data, stdevGaussian):
    #method computes pairwise affinities which is P(j|i) of data when stdevGaussian is the st dev inputted
    #returns numpy matrix where the ith row and jth column is P(j|i)
    
    
    
    numSamples = data.shape[0]
    toReturn = np.zeros((numSamples,numSamples))
    pairWiseDistances = computePairwiseDistances(data)
    
    for i in range(numSamples):
        for j in range(numSamples):
            if(i==j):
                toReturn[i,j] = 0
            else:
                toReturn[i,j] = math.exp( -1 * pairWiseDistances[i,j]**2 / (2 * stdevGaussian**2) )
                
    #normalize to each row total
    row_sums = np.sum(toReturn, axis=1)
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = toReturn[i,j] / row_sums[i]

#     row_totals = np.zeros(range(numSamples))
#     for i in range(len(row_totals)):
#         for k in range(numSamples):
#             if(i != k):
#                 total += math.exp( pairWiseDistances[i,k] / (2 * stdevGaussian**2) )
            
    return toReturn

def getPerplexity(data):
    #returns the perplexity of a given pariwise affinity data matrix
    #TODO need to make per point?
    numSamples = data.shape[0]
    
    toReturn = np.zeros((numSamples))
    
    for i in range(numSamples):
        total = 0
        for j in range(numSamples):
            if(i!=j):
                total += data[i,j] * math.log2(data[i,j])
        toReturn[i] = -1 * total
        
    return toReturn

def symmetrisizePairWiseAffinities(data):
    #makes pairwise affinities symmetric (Pij = pj|i + pi|j /2n)
    #input the pairwise affinities and returns pairwise affinities symmetrisized
    numSamples = data.shape[0]
    
    toReturn = np.zeros((numSamples, numSamples))
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = (data[i,j] + data[j,i]) / (2 * numSamples)
            
    return toReturn


# In[6]:


def computeLowDimAffinities(lowDimData):
    #outputs computes low dimensional affinities (qij) as nxn np matrix
    #input the low dimensional data (assume 2d array of measurements), where each row is an observation and there are 2 columns
    
    numSamples = lowDimData.shape[0]
    toReturn = np.zeros((numSamples,numSamples))
    
    pairWiseDistances = computePairwiseDistances(lowDimData)
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = (1 + (pairWiseDistances[i,j])**2 )**(-1)
            
    #normalize each to total of whole matrix
    total = 0
    for i in range(numSamples):
        for j in range(numSamples):
            if(i != j):
                total += toReturn[i,j]
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = toReturn[i,j] / total
            
            
    return toReturn

def computeGradientforIthPoint(q, p, lowDimData, i):
    #computes the gradient of KL divergence dC/dy between p (high dimensional data affinities)
    #and q (low dim. data affinities)
    #lowDimData is the low dimensional data (2d list) and i is the ith point
    
    toReturnX = 0
    toReturnY = 0
    numSamples = q.shape[0]
    pairWiseDistances = computePairwiseDistances(lowDimData)
    
    for j in range(numSamples):
        toReturnX += (p[i,j]-q[i,j]) * (lowDimData[i,0] - lowDimData[j,0]) * ((1 + (pairWiseDistances[i,j])**2)**-1)
        toReturnY += (p[i,j]-q[i,j]) * (lowDimData[i,1] - lowDimData[j,1]) * ((1 + (pairWiseDistances[i,j])**2)**-1)
    
    
    toReturnX = 4 * toReturnX
    toReturnY = 4 * toReturnY
    return toReturnX, toReturnY
    
def calcKLDivergence(q, p):
    #computes the KL Divergence  between p (high dimensional data affinities) and q (low dim. data affinities)
    total = 0
    numSamples = q.shape[0]
    for i in range(numSamples):
        for j in range(numSamples):
            if(i!=j):
                total += p[i,j] * math.log(p[i,j]/q[i,j]) / math.log(10)
    
    return total
    


# In[7]:


# q = symmetrisizePairWiseAffinities(computePairwiseAffinities(data,10))
# print("perplexity" + str(getPerplexity(computePairwiseAffinities(data,10))))
# lowDimData = twodto2dtSNE(8, colors)
# print(lowDimData)
# p = computeLowDimAffinities(lowDimData)
# computeGradientforIthPoint(q,p,lowDimData,0)

#doTIterations(q,p,lowDimData,2)


# In[8]:


def do1UpdatesOnIth(q, p, lowDimData, indx):
    #does an update on the ith char for lowDimData as 2d with rows being data and 2 columns
    
    numSamples = lowDimData.shape[0]
    updatedData1 = np.zeros((numSamples,numSamples))         
    
    numSamples = q.shape[0]
    gradX, gradY = computeGradientforIthPoint(q, p, lowDimData, indx)
    
    for i in range(len(lowDimData)):
        for j in range(2):
            updatedData1[i,j] = lowDimData[i,j]
    
    updatedData1[indx, 0] = updatedData1[indx,0] - gradX
    updatedData1[indx, 1] = updatedData1[indx,1] - gradY
    
    newQ = computeLowDimAffinities(np.array(updatedData1))
    #print(str(gradX) + ", " + str(gradY))
    
    return updatedData1
    
def doTIterations(q, p, lowDimData, T):
    #runs T iterations and prints out KL Divergence respectively
    
    newData = lowDimData
    newQ = q
    for t in range(T):
        for ith in range(lowDimData.shape[0]):
            newData = do1UpdatesOnIth(newQ,p,newData,ith)
            newQ = computeLowDimAffinities(np.array(newData))
            
    return newData


# In[9]:


def runTSNE(data, colors, stdev, iterations):
    #runs entire tSNE on data with the given st dev
    pairwiseAff = computePairwiseAffinities(data, stdev)
    
    print("Calculated Perplexity: " + str(getPerplexity(pairwiseAff)))
    plotRandom = twodto2dtSNE(data.shape[0],colors)
    q = computeLowDimAffinities(plotRandom)
    p = symmetrisizePairWiseAffinities(pairwiseAff)
    
    updatedLine = doTIterations(q,p,plotRandom,iterations)
    plt.scatter(updatedLine[:,0], updatedLine[:,1], c=colors)
    plt.show()
    return q,p,updatedLine, plotRandom
    


# In[10]:


q,p,updatedLine, onNumberLineRandom = runTSNE(data, colors, 8, 200)
print(updatedLine)


# In[11]:


print(calcKLDivergence(computeLowDimAffinities(np.array(updatedLine)),p))


# In[12]:


computeGradientforIthPoint(q,p,updatedLine,7)


# In[13]:


computeGradientforIthPoint(q,p,updatedLine,4)


# In[ ]:


# Set the number of clusters and the number of points per cluster
num_clusters = 4
points_per_cluster = 4

# Set the dimensionality of each data point
dimensionality = 5

# Set the random seed for reproducibility
np.random.seed(42)

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


# In[ ]:


print(data_matrix.shape[0])


# In[ ]:


#colors = ["red"]*4 + ["blue"]*4 + ["green"]*4 + ["orange"]*4
print(colors)
q,p,updatedLine, onNumberLineRandom = runTSNE(data_matrix, colors, 15, 200)
print(updatedLine)


# In[ ]:





# In[ ]:




