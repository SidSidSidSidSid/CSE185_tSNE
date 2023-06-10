#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import anndata as ad


# In[2]:


def generateInitialtSNE(numData, colors):    
    """ Generates the Initial, Random Points for tSNE and plots on a 2D graph
    
    Parameters
    ----------
    numData : int
       number of data points to generate
    colors : list
       list of string colors for each data point
       
    Returns
    -------
    dataLocs : 2d numpy array where each row is a low dimensional data point. The 2 columns
        represent X and Y values for the initial data
    """
    dataLocs = np.zeros((numData,2))
    for i in range(len(dataLocs)):
        for j in range(2):
            dataLocs[i,j] = np.random.rand()
        
    print("Initial Data: ")
    
    plt.scatter(dataLocs[:,0], dataLocs[:,1], c=colors)
    plt.show()
    print()
    return(dataLocs)


# In[3]:


def computePairwiseDistance(list1, list2):
    """ Computes the pairwise Euclidian distance of 2 lists
    
    Parameters
    ----------
    list1 : list
       list of floating point numbers
    list2 : list
       list of floating point numbers
       
    Returns
    -------
    EuclidianDistance : Euclidian distance between the 2 lists
    """
    
    if(len(list1)!=len(list2)):
        print("ERROR! lists must be same length")
        return None
    
    sumOfSquares = 0
    for i in range(len(list1)):
        sumOfSquares += (list1[i]-list2[i])**2
    return math.sqrt(sumOfSquares)
    
    

def computePairwiseDistances(data):
    """ Computes the pairwise Euclidian distance of 2 lists
    
    Parameters
    ----------
    data : 2D Numpy Array
        2D Numpy Array where rows are observations and columns are values for that observation
       
    Returns
    -------
    EuclidianDistances: Euclidian Distance matrix where the ith row and jth column represents the euclidian distance
        of the ith point to the jth point
    """
    
    return cdist(data, data, metric='euclidean')


# In[4]:


def computePairwiseAffinities(data, stdevsGaussian, pairwiseDistances):
    """ Computes the pairwise affinities (P(j|i)) of the data. This is the probability that the high dimensional point i
        would pick the jth point, if neighbors were chosen in proportion to their probability density under a Gaussian
        distribution centered at the ith point.
    
    Parameters
    ----------
    data : 2D Numpy Array
        2D Numpy Array where rows are observations and columns are values for that observation
    stdevsGaussian : list
       list of standard deviations where the ith standard deviation is the standard deviation used for the ith point
    pairwiseDistances : 2D Numpy Array
        2D Numpy Array where the element at (i,j) is the euclidian distance between the ith and jth point
        
    Returns
    -------
    PairwiseAffinities: 2D Numpy Array where the ith row and jth column is P(j|i)
    """
    
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
    """ Returns the calculated perplexity from a given list of pairwise affinities for the ith point
    
    Parameters
    ----------
    pairwiseAff : list
        2D Numpy Array where rows are observations and columns are values for that observation
    i : int
       list of standard deviations where the ith standard deviation is the standard deviation used for the ith point

    Returns
    -------
    perplexity: calculated perplexity
    """
    
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
    """ Returns the calculated perplexity of the ith point from the given data and standard deviation, with the 
    Pairwise distance matrix already calculated
    
    Parameters
    ----------
    data : 2D Numpy Array
        2D Numpy Array where rows are observations and columns are values for that observation
    stdevGaussian : floating point number
        A Gaussian standard deviation to used to calculate the perplexity. As this increased, perplexity also increases.
    i : int
       The ith point to calculate the perplexity for
    pairwiseDistances : 2D Numpy Array
        2D Numpy Array where the (i,j) point is the distance between the ith and jth data point

    Returns
    -------
    perplexity: calculated perplexity
    """
    
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


# In[5]:


def computeStDevGaussianOfIth(data, perplexity, i, startingLowerBound, startingUpperBound, allowedError, pairwiseDistances):
    """ Computes the Gaussian Standard Deviation of the ith point using binary search for a given perplexity
    
    Parameters
    ----------
    data : 2D Numpy Array
        2D Numpy Array where rows are observations and columns are values for that observation
    perplexity : floating point number
        The expected number of neighbors that point is expected to have
    i : int
       The ith point to calculate the gaussian standard deviation for
    startingLowerBound : floating point number
        The lower bound to start binary search with
    startingUpperBound : floating point number
        The upper bound to start binary search with
    allowedError : floating point number
        The amount of error in perplexity this algorithm allows.
    pairwiseDistances : 2D Numpy Array
        2D Numpy Array where the (i,j) point is the distance between the ith and jth data point

    Returns
    -------
    StandardDeviationGaussian : The calculated Gaussian Standard Deviation for the given perplexity using binary search.
    """
    
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
    
    print("ERROR in computeStDevGaussianOfIth for the " + str(i) + " point", flush=True)
    toReturn = getPerplexityOfIthPointFromData(data, (lower+upper)/2, i, pairwiseDistances)
    print("Wrongly using " + str(toReturn) + " as the calculated perplexity for this data point", flush=True)
    return (lower+upper)/2

def getListOfStDevGaussian(data, perplexity, pairwiseDistances):
    """ Computes a list of Gaussian Standard Deviation using binary search for a given perplexity
    
    Parameters
    ----------
    data : 2D Numpy Array
        2D Numpy Array where rows are observations and columns are values for that observation
    perplexity : floating point number
        The expected number of neighbors that point is expected to have
    pairwiseDistances : 2D Numpy Array
        2D Numpy Array where the (i,j) point is the distance between the ith and jth data point

    Returns
    -------
    StandardDeviationGaussian : The calculated Gaussian Standard Deviation for the given perplexity using binary search as 
        a list for every point.
    """
    
    numSamples = data.shape[0]
    toReturn = np.zeros((numSamples))
    
    count = 0
    
    for i in range(numSamples):
        count += 1
        if(count % 50 == 0):
            print(str(count) + " done for st dev gaussian", flush = True)
        toReturn[i] = computeStDevGaussianOfIth(data, perplexity, i, 0, 999, perplexity*0.1, pairwiseDistances)
        
    return toReturn


# In[6]:


def getPerplexity(pairwiseAff):
    """ Computes a list of perplexities, given the pairwise affinities (before symmetrization)
    
    Parameters
    ----------
    pairwiseAff : 2D Numpy Array
        2D Numpy Array where the ith row and jth column is P(j|i). This is the probability that the high dimensional point i
        would pick the jth point, if neighbors were chosen in proportion to their probability density under a Gaussian
        distribution centered at the ith point.

    Returns
    -------
    PerplexityList : The calculated perplexity for each data point as a list
    """
    
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

def symmetrisizePairWiseAffinities(pairwiseAff):
    """ Symmetries the pairwise affinities. (Pij = pj|i + pi|j /2n)
    
    Parameters
    ----------
    pairwiseAff : 2D Numpy Array
        2D Numpy Array where the ith row and jth column is P(j|i). This is the probability that the high dimensional point i
        would pick the jth point, if neighbors were chosen in proportion to their probability density under a Gaussian
        distribution centered at the ith point.

    Returns
    -------
    symmetrisizedPairWiseAffinities : 2D numpy array of the pairwise affinities being symmetric now
    """
    
    numSamples = pairwiseAff.shape[0]
    
    toReturn = np.zeros((numSamples, numSamples))
    
    for i in range(numSamples):
        for j in range(numSamples):
            toReturn[i,j] = (pairwiseAff[i,j] + pairwiseAff[j,i]) / (2 * numSamples)
            
    return toReturn


# In[7]:


def computeLowDimAffinities(lowDimData):
    """ Computes the low dimensional pairwise Affinities. Q(j|i) is the probability that the low dimensional point i
        would pick the jth point, if neighbors were chosen in proportion to their probability density under a student's T
        distribution centered at the ith point.
    
    Parameters
    ----------
    lowDimData : 2D Numpy Array
        2D numpy array where each row is a sample and there are 2 columns representing X and Y values.

    Returns
    -------
    lowDimAffinities : 2D nxn numpy array, where (i,j) is Q(j|i)
    """
    
    toReturn = computePairwiseDistances(lowDimData)
    toReturn = toReturn**2
    toReturn = toReturn + 1
    toReturn = toReturn ** (-1)

    total = np.sum(toReturn)
    
    toReturn = toReturn / total
            
            
    return toReturn

def computeGradientforIthPoint(q, p, lowDimData, i):
    """ Computes the gradient of KL divergence dC/dy between p (high dimensional data affinities)
        and q (low dim. data affinities)
    
    Parameters
    ----------
    q : 2D Numpy Array
        Low Dimensional Affinity Matrix
    p : 2D Numpy Array
        High Dimensional Affinity Matrix
    lowDimData : 2D Numpy Array
        2D numpy array where each row is a sample and there are 2 columns representing X and Y values.
    i : int
            Gradient is calculated for the ith point


    Returns
    -------
    GradX : floating point number representing how the ith point should be moved on the X axis to minimize the gradient
    GradY: floating point number representing how the ith point should be moved on the Y axis to minimize the gradient
    """
    
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
    """ Computes the Kullback-Leibler Divergence between the low dimensional affinities and the high 
    dimensional affinities
    
    Parameters
    ----------
    q : 2D Numpy Array
        Low Dimensional Affinity Matrix
    p : 2D Numpy Array
        High Dimensional Affinity Matrix

    Returns
    -------
    Divergence : floating point number representing the Kullback-Leibler Divergence
    """

    total = 0
    numSamples = q.shape[0]
    for i in range(numSamples):
        for j in range(numSamples):
            if(i!=j):
                total += p[i,j] * math.log(p[i,j]/q[i,j]) / math.log(10)
    
    return total
    


# In[8]:


def do1UpdatesOnIth(q, p, lowDimData, indx):
    """ Does an update on the indx data point in the low dimensional space to minimize the Kullback-Leibler Divergence
    
    Parameters
    ----------
    q : 2D Numpy Array
        Low Dimensional Affinity Matrix
    p : 2D Numpy Array
        High Dimensional Affinity Matrix
    lowDimData : 2D Numpy Array
        2D numpy array where each row is a sample and there are 2 columns representing X and Y values.
    indx : int
            The indx element to do an update for


    Returns
    -------
    lowDimData : new low dimensional data.
    newQ: new high dimensional affinity matrix for this new low dimensional data
    """     
    
    gradX, gradY = computeGradientforIthPoint(q, p, lowDimData, indx)

    lowDimData[indx, 0] = lowDimData[indx,0] - gradX
    lowDimData[indx, 1] = lowDimData[indx,1] - gradY
    
    newQ = computeLowDimAffinities(np.array(lowDimData))
    
    return lowDimData, newQ
    
def doTIterations(q, p, lowDimData, T, printUpdates=False):
    """ Does T iteration updates on the low dimensional data. Each iteration updates every data point once.
    
    Parameters
    ----------
    q : 2D Numpy Array
        Low Dimensional Affinity Matrix
    p : 2D Numpy Array
        High Dimensional Affinity Matrix
    lowDimData : 2D Numpy Array
        2D numpy array where each row is a sample and there are 2 columns representing X and Y values.
    T : int
            number of iterations
    printUpdates : boolean
        Whether updates should be printed for every other iteration.


    Returns
    -------
    lowDimData : new low dimensional data.
    """
    
    newData = lowDimData
    newQ = q
    for t in range(T):
        if(printUpdates and (t+1)%2 == 0):
            print("Iteration " + str(t+1) + " started", flush=True)
        for ith in range(lowDimData.shape[0]):
            newData, newQ = do1UpdatesOnIth(newQ,p,newData,ith)
            
    return newData


# In[9]:


def runTSNE(data, perplexity, iterations, numPCs = 0, colors = None):
    """ Runs the TSNE algorithm, printing the calculated perplexity for each data point, plotting the initial data, and 
    plotting the final data after tSNE.
    
    Parameters
    ----------
    data : 2D Numpy Array
        2D Numpy Array where rows are observations and columns are values for that observation
    perplexity : int
       the expected number of neighbors each data point is expected to have. Usually, this is between 5 and 50.
    iterations : 2D Numpy Array
        The number of iterations for myTSNE to run through.
    numPCs : int
        The number of PCs to calculate beforehand if necessary to speed up computation. If left blank, the default is 0. 
        If 0 is inputted, PCs are not calculated. Usually, this is between 10 and 50.
    colors : list
        A list with length of the number of rows in data that represent matplotlib colors to plot the data. If left blank,
        the default is None and all points will be plotted as gray.

    Returns
    -------
    lowDimData : The low dimensional result of tSNE as a 2D Numpy Array. Each row is a different data point, and the 1st
        column is the x value and the 2nd column is the y value.
    """
    
    pairwiseAff = None
    pairwiseDistances = None
    
    if(numPCs <= 0):
        print("Top PCs are not used", flush=True)
        pairwiseDistances = computePairwiseDistances(data)
        pairwiseAff = computePairwiseAffinities(data, getListOfStDevGaussian(data, perplexity, pairwiseDistances), pairwiseDistances)
    else:
        print("Calculating PCs", flush=True)
        pca = PCA(n_components=numPCs)
        projected_data = pca.fit_transform(data)
        pairwiseDistances = computePairwiseDistances(projected_data)
        pairwiseAff = computePairwiseAffinities(projected_data, getListOfStDevGaussian(projected_data, perplexity, pairwiseDistances), pairwiseDistances)
    
    print("Finished Calculating Pairwise Affinities in High Dimension", flush=True)
    
    print("Calculated Perplexity List: " + str(getPerplexity(pairwiseAff)), flush=True)
    plotRandom = generateInitialtSNE(data.shape[0],colors)
    q = computeLowDimAffinities(plotRandom)
    p = symmetrisizePairWiseAffinities(pairwiseAff)
    
    lowDimData = doTIterations(q,p,plotRandom,iterations, printUpdates = True)
    if(colors is None):
        colors = ["gray"] * data.shape[0]
    
    plt.scatter(lowDimData[:,0], lowDimData[:,1], c=colors)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Plot')
    
    plt.show()
    q = computeLowDimAffinities(lowDimData)
    divergence = calcKLDivergence(q,p)
    print("Divergence: " + str(divergence), flush=True)
    return lowDimData
    


# In[10]:


def colorByGene(highDimData, lowDimData, geneName, annDataGeneNames):
    """ Makes the tSNE plot colored by genes
    
    Parameters
    ----------
    highDimData : 2D Numpy Array
        2D numpy array where each row is a sample and each column a gene
    lowDimData : 2D Numpy Array
        2D numpy array where each row is a sample and there are 2 columns representing X and Y values.
    geneName : string
        string name of the gene to be colored by
    annDataGeneNames : list
        list of strings for all the gene names in order as highDimData

    Returns
    -------
    colors : The colors as a list of each cell in order.
    """
    
    indexOfGene = -1
    
    for i in range(len(annDataGeneNames)):
        if(geneName == annDataGeneNames[i]):
            indexOfGene = i
            
    if(indexOfGene == -1):
        print("Gene Name not found")
        return None
    
    values = highDimData[:, indexOfGene]
    
    colormap = plt.cm.get_cmap('coolwarm')
    
    # Normalize the values between 0 and 1
    norm = plt.Normalize(min(values), max(values))
    normalized_values = norm(values)
    colors = [colormap(value) for value in normalized_values]
    
    plt.scatter(lowDimData[:,0], lowDimData[:,1], c=colors)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Plot colored by ' + str(geneName))
    plt.show()
    
    return colors

