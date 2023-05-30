# CSE185_tSNE Description
My name is Siddharth Gaywala and this is my version of the ReadMe made for the CSE185 T-SNE project proposal. T-SNE is a way to visualize high dimensional data in lower dimensions. In the algorithm presented, high dimensional data can be visualized in two dimensions. I currently have some base code and some example data to test it on.

Base code for the tSNE algorithm can be found in mytSNE/mytSNE_052923.py. My sample data can be found in mytSNE/tSNE_sample_data 052923.py.

This tSNE algorithm works by plotting all points randomely onto a 2D graph. Pairwise probabilities that the ith point would pick the jth point as its neighbor are calculated for the low dimensional data and the high dimensional data. Then, points are moved one-by-one in the direction to minimize the difference between the pairwise probabilities between the low dimensional and high dimensional data (using a cost function using gradient descent).

## Installation Instructions
To install my tSNE algorithm, place the mytSNE_052923.py in the same folder as your other python file. Then, run the following command:
```
import mytSNE_052923
```

To run my tSNE algorithm in the mytSNE_052623.py file, call the mytSNE_052923.runTSNE() method, which takes in 5 parameters and has 1 output.
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

This data looks like if plotted on a 2d scale:
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
#data is as follows
[[ 16.05161808  -7.0051931   -6.02887582  -9.03723162   8.70488405]
 [ 15.60645799  -5.92664865  -3.18146239 -10.60952727   9.2712794 ]
 [ 16.54362396  -6.46981398  -6.42423572 -11.07902894   8.44518206]
 [ 16.83007683  -5.27858072  -4.35061544 -10.4440989    9.53921746]
 [-23.76978491  18.7009858   -7.09913919   2.89229813  -2.00518561]
 [-23.09095868  18.57974703  -6.09225219   5.37596637  -3.89020009]
 [-24.45950077  16.94365178  -7.45203194   4.06655988  -2.17806881]
 [-25.03758818  17.14191363  -6.78409437   3.4204857   -1.73169257]
 [ 14.39875123 -20.80216516  -3.03761065  -3.4304919   11.53599414]
 [ 14.74008802 -21.27206938  -2.84660825  -3.71872228  12.46717833]
 [ 15.81999725 -20.41625068  -3.59945699  -4.47927395  11.76118878]
 [ 14.69841944 -20.94526077  -3.18057518  -4.46054439  12.03572646]
 [-11.44604124  -0.49977437  -8.37509254   1.01571599   4.73324029]
 [-10.82953024  -0.98372562  -9.73228478   0.15591896   5.86076668]
 [-12.37202999  -1.40912268  -7.93242353  -0.43737847   6.17869812]
 [-12.31119608  -1.76297758 -10.39435653   1.54355518   6.23705268]]
```

I ran the following commands:
```
colors = ["red"]*4 + ["blue"]*4 + ["green"]*4 + ["orange"]*4
lowDimData = mytSNE_052923.runTSNE(data_matrix, colors, perplexity=3, iterations=75, numPCs = 0)
```

The output of myTSNE looks like this:
![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/4f4a1649-1b47-45e7-ba48-e47cb2c6f9c3)

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/70341260-26a5-4b22-9b3d-158d8e175a2e)

