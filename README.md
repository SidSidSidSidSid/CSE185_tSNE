# CSE185_tSNE Description
My name is Siddharth Gaywala and this is my version of the ReadMe made for the CSE185 T-SNE project proposal. T-SNE is a way to visualize high dimensional data in lower dimensions. In the algorithm presented, high dimensional data can be visualized in two dimensions. I currently have some base code and some example data to test it on. The algorithm described is based on the algorithm from [this paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

Base code for the tSNE algorithm can be found in mytSNE/mytSNE_060823.py. My sample data can be found in mytSNE/tSNE_sample_data_060823.py.

This tSNE algorithm works by plotting all points randomely onto a 2D graph. Pairwise probabilities that the ith point would pick the jth point as its neighbor are calculated for the low dimensional data and the high dimensional data. Then, points are moved one-by-one in the direction to minimize the difference between the pairwise probabilities between the low dimensional and high dimensional data (using a cost function using gradient descent).

## Installation Instructions
To install my tSNE algorithm, place the mytSNE_060823.py in the same folder as your other python file. Then, run the following command in python:
```
import mytSNE_060823 as mytSNE
```

To run my tSNE algorithm in the mytSNE_060823.py file, call the mytSNE_060823.runTSNE() method, which takes in 5 parameters and has 1 output.
Parameters are as follows:
1. data: a 2d numpy array where rows represent each sample and columns represents observations for that sample.
2. perplexity: the expected number of neighbors each data point is expected to have. Usually, this is between 5 and 50.
3. iterations: The number of iterations for myTSNE to run through. 
4. numPCs: The number of PCs to calculate beforehand if necessary to speed up computation. If left blank, the default is 0. If 0 is inputted, PCs are not calculated. Usually, this is between 10 and 50.
5. colors: a list with length of the number of rows in data that represent matplotlib colors to plot the data. If left blank, the default is None. If None is inputted, colors of all points will be gray.

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
colors = np.array(["forestgreen", "limegreen","darkgreen","lime", "navy", "slateblue","lavender","royalblue","maroon","tomato","salmon","lightcoral",])
```

If plotted on a 2d scale, this data looks like:
![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/ccda8a2f-d5ef-4a77-9a98-eb37aff5e8e3)


Then, you can run the following command to run myTSNE on this data with a perplexity of 8 and 30 iterations.
```
lowDimData = mytSNE.runTSNE(data, 8, 30, 0, colors)
```
This is the output:

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/add7cd19-a879-4fb0-ab44-8c1d306a2731)

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/7b31b776-976a-4bba-84be-f26c965a6580)

## Sample Code on a More Complex Example

The data inputted consists of 5 dimensional data made of 4 clusters.
```
import numpy as np
#data_matrix is as follows
data_matrix = np.array([[ 15.66794698,  -8.78045103,  -7.5231924 ,  -5.65232242, 8.80649956],
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
lowDimData = mytSNE.runTSNE(data_matrix, perplexity=3, iterations=75, numPCs = 0, colors
```

The output of myTSNE looks like this:

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/4f4a1649-1b47-45e7-ba48-e47cb2c6f9c3)

![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/70341260-26a5-4b22-9b3d-158d8e175a2e)


## Running on scRNA-seq Data
Steps:
1. Generate an anndata variable and perform filtering for genes and cells. (For this algorithm, a max of 300 cells is recommended for faster computation).
```
import anndata as ad
#generate an anndata variable called adata_var and perform filtering on it
```
2. Run the tSNE algorithm
```
import mytSNE_060823 as mytSNE
import numpy as np
highDimData = np.array(adata_var.X.todense())
lowDimData= mytSNE.runTSNE(highDimData, perplexity, iterations, numPCs)
```
3. Color by Gene (if necessary)
The function mytSNE.colorByGene() takes in 4 parameters and outputs the list of colors as well as a matplotlib plot of the tSNE data colored by those colors.
The 4 parameters are:
- highDimData: high dimensional data
- lowDimData: low dimensional data generated after running mytSNE
- geneName: name of the gene you want to color by
- annDataGeneNames: gene names of the genes
```
colors = mytSNE.colorByGene(highDimData, lowDimData, "{marker name}", adata_var.var_names)
```

## Sample Code on scRNA-seq Data
This is a dataset taken from [this paper](https://www.nature.com/articles/s41591-020-0901-9). Fluid was collected from lungs of patients with severe and mild COVID-19 and scRNA-seq was performed on that fluid.

```
# load data into an anndata variable called adata_control for a control sample and adata_severe for a COVID-patient sample
# adata_control consisted of 544 cells & 503 genes (GEO accession number GSE145926, sample GSM4339769_C141))
# adata_severe consisted of 319 cells & 504 genes (GEO accession number GSE145926, sample GSM4339773_C145)
controlHighDimData = np.array(adata_control.X.todense())
severeHighDimData = np.array(adata_severe.X.todense())

controlLowDimData = mytSNE.runTSNE(np.array(adata_control.X.todense()), perplexity=25, iterations=1000, numPCs = 10)
severeLowDimData = mytSNE.runTSNE(np.array(adata_severe.X.todense()), perplexity=25, iterations=1000, numPCs = 10)
```
Then, I generated the appropriate graphs coloring by CD68 (marker for macrophages), FABP4, SPP1, and FCN1.
```
colorsCD68_control = mytSNE.colorByGene(controlHighDimData, controlLowDimData, "CD68", adata1_var.var_names)
colorsFABP4_control = mytSNE.colorByGene(controlHighDimData, controlLowDimData, "FABP4", adata1_var.var_names)
colorsSPP1_control = mytSNE.colorByGene(controlHighDimData, controlLowDimData, "SPP1", adata1_var.var_names)
colorsFCN1_control = mytSNE.colorByGene(controlHighDimData, controlLowDimData, "FCN1", adata1_var.var_names)

colorsCD68_severe = mytSNE.colorByGene(severeHighDimData, severeLowDimData, "CD68", adata1_var.var_names)
colorsFABP4_severe = mytSNE.colorByGene(severeHighDimData, severeLowDimData, "FABP4", adata1_var.var_names)
colorsSPP1_severe = mytSNE.colorByGene(severeHighDimData, severeLowDimData, "SPP1", adata1_var.var_names)
colorsFCN1_severe = mytSNE.colorByGene(severeHighDimData, severeLowDimData, "FCN1", adata1_var.var_names)
```


| Marker      | Patient with Mild COVID-19 | Patient with Severe COVID-19 |
| ----------- | ----------- | -----|
| CD68      | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/22f1d448-934f-4121-95d5-ed3f77d6346b)       | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/3d63b905-80b0-49b3-93ca-2b4d9dfafb0c) |
| FABP4   | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/4fdd5d10-37a2-4c07-9f1e-e23d197384c2)        | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/fb56d1ab-e1d8-4ca3-ba3f-c5a21a61ce10) |
| SPP1   | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/ef39cdec-d1cf-4d51-a1de-29bfa4779739)        | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/3b24ce95-0f55-4879-bd18-3cefd7a51118) |
| FCN1   | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/7566742b-4029-4ee4-b397-dfcf51c16ea5)        | ![image](https://github.com/Siddharth-Gaywala/CSE185_tSNE/assets/38893705/2ce94888-f9a3-4b35-bbb2-342c0f836bc0) |

This matches several of the paper findings, where macrophage clusters tended to be enriched for SPP1 and FCN1 in patients with severe COVID and macrophage clusters tended to be enriched for FABP4 in healthy control patients.
