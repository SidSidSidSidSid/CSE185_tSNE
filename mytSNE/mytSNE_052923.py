#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import sys

def twodto2dtSNE(numData, colors):    
    dataLocs = np.zeros((numData,2))
    for i in range(len(dataLocs)):
        for j in range(2):
            dataLocs[i,j] = np.random.rand()
        
    print("Initial Data: ")
    
    plt.scatter(dataLocs[:,0], dataLocs[:,1], c=colors)
    plt.show()
    print()
    return(dataLocs)


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
    return cdist(data, data, metric='euclidean')


# In[3]:


def computePairwiseAffinities(data, stdevsGaussian, pairwiseDistances):
    #method computes pairwise affinities which is P(j|i) of data when stdevGaussian is the st dev inputted as a list
    #returns numpy matrix where the ith row and jth column is P(j|i)
    
    
    
    numSamples = data.shape[0]
    toReturn = np.zeros((numSamples,numSamples))
    
    for i in range(numSamples):
        for j in range(numSamples):
            if(i==j):
                toReturn[i,j] = 0
            else:
                toReturn[i,j] = math.exp( -1 * pairwiseDistances[i,j]**2 / (2 * stdevsGaussian[i]**2) )
                
    #normalize to each row total
    row_sums = np.sum(toReturn, axis=1)
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = toReturn[i,j] / row_sums[i]
            
    return toReturn

def getPerplexityOfIthPointFromAffinities(pairwiseAff, i):
    #returns the perplexity of the ith point from a given list of pairwise affinities (subset of 2d pairwise affinities matrix)

    
    numSamples = len(pairwiseAff)
    
    toReturn = 0
    
    for j in range(numSamples):
        if(i!=j):
            if(pairwiseAff[j] == 0):
                #add pseudocount
                pairwiseAff[j] = 1e-20
            toReturn += pairwiseAff[j] * math.log2(pairwiseAff[j])
            
    toReturn = -1 * toReturn
    toReturn = 2 ** toReturn
    
    return toReturn


def getPerplexityOfIthPointFromData(data, stdevGaussian, i, pairwiseDistances):
    #returns the perplexity of the ith point from a given data list and stdevGaussian
    
    numSamples = data.shape[0]
    pairwiseAff = np.zeros((numSamples))
    
    
    for j in range(numSamples):
        if(i==j):
            pairwiseAff[j] = 0
        else:
            pairwiseAff[j] = math.exp( -1 * pairwiseDistances[i,j]**2 / (2 * stdevGaussian**2) )
            if(pairwiseAff[j] == 0):
                #add pseudocount
                pairwiseAff[j] = 1e-20
                
    
    sumPairWise = np.sum(pairwiseAff)
    for j in range(numSamples):
        pairwiseAff[j] = pairwiseAff[j] / sumPairWise
        
    
    
    return getPerplexityOfIthPointFromAffinities(pairwiseAff, i)


# In[4]:


def computeStDevGaussianOfIth(data, perplexity, i, startingLowerBound, startingUpperBound, allowedError, pairwiseDistances):
    #given data input, a given perplexity, and the ith point, calculates the ideal guassian standard deviation 
    # using binary search
    numSamples = data.shape[0]
    
    
    #performs binary search
    lower = startingLowerBound
    upper = startingUpperBound
    while(lower < upper):
        middle = (lower+upper)/2
        calcMiddlePerp = getPerplexityOfIthPointFromData(data, middle, i, pairwiseDistances)
        
        if(abs(calcMiddlePerp - perplexity) <= allowedError ):
            return middle
        elif(perplexity < calcMiddlePerp):
            #upper = middle - allowedError/2
            upper = middle - 1e-20
        else:
            #lower = middle + allowedError/2
            lower = middle + 1e-20
    
    print("ERROR in computeStDevGaussianOfIth")
    sys.exit()
    return None

def getListOfStDevGaussian(data, perplexity, pairwiseDistances):
    #returns a list of Standard Deviations Gaussian for a given perplexity
    numSamples = data.shape[0]
    toReturn = np.zeros((numSamples))
    
    count = 0
    
    for i in range(numSamples):
        count += 1
        if(count % 50 == 0):
            print(str(count) + " done for st dev gaussian", flush = True)
        toReturn[i] = computeStDevGaussianOfIth(data, perplexity, i, 0, 999, perplexity*0.1, pairwiseDistances)
        
    return toReturn


# In[5]:


def getPerplexity(pairwiseAff):
    #returns the perplexity of a given pariwise affinity data matrix
    numSamples = pairwiseAff.shape[0]
    
    toReturn = np.zeros((numSamples))
    
    for i in range(numSamples):
        total = 0
        for j in range(numSamples):
            if(i!=j):
                if(pairwiseAff[i,j]==0):
                    pairwiseAff[i,j] = 1e-20
                total += pairwiseAff[i,j] * math.log2(pairwiseAff[i,j])
        toReturn[i] = -1 * total
        toReturn[i] = 2 ** toReturn[i]
        
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
    
    
    toReturn = computePairwiseDistances(lowDimData)
    toReturn = toReturn**2
    toReturn = toReturn + 1
    toReturn = toReturn ** (-1)

    total = np.sum(toReturn)
    
    toReturn = toReturn / total
            
            
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


def do1UpdatesOnIth(q, p, lowDimData, indx):
    #does an update on the ith char for lowDimData as 2d with rows being data and 2 columns
    
#     numSamples = lowDimData.shape[0]
#     updatedData1 = np.zeros((numSamples,numSamples))         
    
    gradX, gradY = computeGradientforIthPoint(q, p, lowDimData, indx)

    lowDimData[indx, 0] = lowDimData[indx,0] - gradX
    lowDimData[indx, 1] = lowDimData[indx,1] - gradY
    
    newQ = computeLowDimAffinities(np.array(lowDimData))
    
    return lowDimData, newQ
    
def doTIterations(q, p, lowDimData, T, printUpdates=False):
    #runs T iterations and prints out KL Divergence respectively
    
    newData = lowDimData
    newQ = q
    for t in range(T):
        if(printUpdates and (t+1)%2 == 0):
            print(str(t+1) + " iteration started", flush=True)
        for ith in range(lowDimData.shape[0]):
            newData, newQ = do1UpdatesOnIth(newQ,p,newData,ith)
            
    return newData


# In[13]:


def runTSNE(data, colors, perplexity, iterations, numPCs = 0):
    #runs entire tSNE on data
    #if numPCs is 0, use all data
    #if numPCs is any other number, runs tSNE on projects of the top numPCs to 
    #speed up computation by reducing dimensionality of dataset
    
    pairwiseAff = None
    pairwiseDistances = None
    
    if(numPCs <= 0):
        pairwiseDistances = computePairwiseDistances(data)
        pairwiseAff = computePairwiseAffinities(data, getListOfStDevGaussian(data, perplexity, pairwiseDistances), pairwiseDistances)
    else:
        print("Calculating PCAs", flush=True)
        pca = PCA(n_components=numPCs)
        projected_data = pca.fit_transform(data)
        pairwiseDistances = computePairwiseDistances(projected_data)
        print("finished calculating pairwise distances", flush=True)
        pairwiseAff = computePairwiseAffinities(projected_data, getListOfStDevGaussian(projected_data, perplexity, pairwiseDistances), pairwiseDistances)
    
    print("finished calculating pairwise aff", flush=True)
    
    print("Calculated Perplexity: " + str(getPerplexity(pairwiseAff)), flush=True)
    plotRandom = twodto2dtSNE(data.shape[0],colors)
    q = computeLowDimAffinities(plotRandom)
    p = symmetrisizePairWiseAffinities(pairwiseAff)
    
    lowDimData = doTIterations(q,p,plotRandom,iterations, printUpdates = True)
    plt.scatter(lowDimData[:,0], lowDimData[:,1], c=colors)
    plt.xlabel('tSNE 1')
    plt.ylabel('tSNE 2')
    plt.title('t-SNE Plot')
    
    plt.show()
    q = computeLowDimAffinities(lowDimData)
    divergence = calcKLDivergence(q,p)
    # print("Divergence: " + str(divergence))
    return lowDimData
    


# In[ ]:




