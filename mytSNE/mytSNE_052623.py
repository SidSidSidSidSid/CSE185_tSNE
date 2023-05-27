#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import math

def twodto2dtSNE(numData, colors):    
    dataLocs = np.zeros((numData,2))
    for i in range(len(dataLocs)):
        for j in range(2):
            dataLocs[i,j] = np.random.rand()
        
    print("Random")
    
    plt.scatter(dataLocs[:,0], dataLocs[:,1], c=colors)
    plt.show()
    return(dataLocs)


# In[9]:


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


# In[10]:


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


# In[11]:


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
    


# In[12]:


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


# In[13]:


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
    


# In[ ]:





# In[ ]:




