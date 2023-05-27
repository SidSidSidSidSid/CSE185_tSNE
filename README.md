# CSE185_tSNE
My name is Siddharth Gaywala and this is my version of the ReadMe made for the CSE185 T-SNE project proposal. I currently have some base code and some example data to test it on.

Base code for the tSNE algorithm can be found in mytSNE/mytSNE_052623.py. My sample data can be found in mytSNE/tSNE_test_sample_data 052623.py.

To run my tSNE algorithm in the mytSNE_052623.py file, call the runTSNE() method, which takes in 4 parameters and has 4 outputs.
Parameters are as follows:
1. data: a 2d numpy array where rows represent each sample and columns represents observations for that sample.
2. colors: a list with length of the number of rows in data that represent matplotlib colors to plot the data.
3. stdev: represents a standard deviation number to calculate perplexities. My code will output a list of perplexities for that given standard deviation number. To increase the perplexity, increase the standard deviation number inputted.
4. iterations: The number of iterations for myTSNE to run through.

Here is sample code:
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

Then, you can run the following command to run myTSNE on this data with a standard deviation of 8 and 200 iterations.
```
q,p,updatedPoints, initialPoints = mytSNE_052623.runTSNE(data, colors, 8, 200)
```
This is the output:
The first line represents the calculated perplexities. The next line (the first graph) is the initial distribution of points made by myTSNE. The last line is the tSNE algorithm plotted after all iterations.
![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/cf76f8e3-ddff-42e3-98a2-7d02f5f8fc4a)

The first and second output of the function is q and p (conditional probabilities of the ith data points if they would be neighbors with the jth datapoint for the low and high dimensional datasets respectively). The third and fourth outputs are the updated points after the algorithm and the initial set of points by the algorithm respectively.

