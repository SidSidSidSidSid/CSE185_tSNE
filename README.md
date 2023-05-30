# CSE185_tSNE Description
My name is Siddharth Gaywala and this is my version of the ReadMe made for the CSE185 T-SNE project proposal. T-SNE is a way to visualize high dimensional data in lower dimensions. In the algorithm presented, high dimensional data can be visualized in two dimensions. I currently have some base code and some example data to test it on.

Base code for the tSNE algorithm can be found in mytSNE/mytSNE_052923.py. My sample data can be found in mytSNE/tSNE_sample_data 052923.py.

This tSNE algorithm works by plotting all points randomely onto a 2D graph. Pairwise probabilities that the ith point would pick the jth point as its neighbor are calculated for the low dimensional data and the high dimensional data. Then, points are moved one-by-one in the direction to minimize the difference between the pairwise probabilities between the low dimensional and high dimensional data (using a cost function using gradient descent).

## Installation Instructions
To install my tSNE algorithm, place the mytSNE_052923.py in the same folder as your other python file. Then, run the following command:
```
import mytSNE_052923
```

To run my tSNE algorithm in the mytSNE_052923.py file, call the mytSNE_052923.runTSNE() method, which takes in 5 parameters and has 1 output.
Parameters are as follows:
1. data: a 2d numpy array where rows represent each sample and columns represents observations for that sample.
2. colors: a list with length of the number of rows in data that represent matplotlib colors to plot the data.
3. perplexity: the expected number of neighbors each data point is expected to have. Usually, this is between 5 and 50.
4. iterations: The number of iterations for myTSNE to run through. 
5. numPCs: The number of PCs to calculate beforehand if necessary to speed up computation. If left blank, the default is 0. If 0 is inputted, PCs are not calculated. Usually, this is between 10 and 50.

Output is a numpy data matrix of the low dimensional data with rows representing each sample, and the 2 columns representing x and y values of the low dimensional data.

## Sample Code on a Simple Example
Here is sample code:
This data consists of 3 clusters in 2 dimensions.
```
import numpy as np
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
```

If plotted on a 2d scale, this data looks like:
![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/5c31c69a-d129-4b67-9183-9758c0923bb4)

Then, you can run the following command to run myTSNE on this data with a perplexity of 3 and 30 iterations.
```
lowDimData = mytSNE_052923.runTSNE(data, colors, 3, 30)
```
This is the output:

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/03d2cfbc-02eb-4249-8eaf-0e21cc6441ea)

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/d74e1fd0-f894-447d-b542-201d355dbd64)

## Sample Code on a More Complex Example

The data inputted consists of 5 dimensional data made of 4 clusters.
```
#data_matrix is as follows
data_matrix2 = np.array([[ 15.66794698,  -8.78045103,  -7.5231924 ,  -5.65232242, 8.80649956],
       [ 14.3324667 ,  -5.54481768,   1.01904789, -10.36920936, 10.50568562],
       [ 17.1439646 ,  -7.17431368,  -8.70927212, -11.77771439, 8.02739359],
       [ 18.00332321,  -3.60061389,  -2.48841128,  -9.87292425, 11.30949979],
       [-25.27858079,  21.20672211,  -6.07327955,   2.29611246, -1.02814932],
       [-23.24210211,  20.8430058 ,  -3.05261856,   9.74711718, -6.68319276],
       [-27.34772839,  15.93472005,  -7.1319578 ,   5.81889772, -1.54679891],
       [-29.08199062,  16.5295056 ,  -5.12814508,   3.88067517, -0.20767021],
       [ 13.95409494, -21.2036813 ,  -2.66448787,  -2.61038861, 11.93259358],
       [ 14.97810531, -22.61339395,  -2.09148068,  -3.47507973, 14.72614615],
       [ 18.21783301, -20.04593784,  -4.35002689,  -5.75673477, 12.60817749],
       [ 14.85309958, -21.63296812,  -3.09338147,  -5.70054608, 13.43179053],
       [-12.34029837,   1.94924104,  -7.56810925,   2.20287304, 2.5434166 ],
       [-10.49076537,   0.49738728, -11.63968598,  -0.37651805, 5.92599578],
       [-15.11826463,  -0.7788039 ,  -6.24010224,  -2.15641036, 6.87979007],
       [-14.93576291,  -1.8403686 , -13.62590124,   3.78639059, 7.05485375]])
colors = ["red"]*4 + ["blue"]*4 + ["green"]*4 + ["orange"]*4
```

Then, I ran the following commands:
```
lowDimData = mytSNE_052923.runTSNE(data_matrix, colors, perplexity=3, iterations=75, numPCs = 0)
```

The output of myTSNE looks like this:

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/4f4a1649-1b47-45e7-ba48-e47cb2c6f9c3)

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/70341260-26a5-4b22-9b3d-158d8e175a2e)

