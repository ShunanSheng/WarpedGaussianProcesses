# Warped Gaussian Process Classfication

## Experiment
Multiple expriments have been conducted to validate our algorithm. In the [test_Synthetic_Data.m][e01], we create a synthetic dataset of sensor network depolyed in R2. Based on the data, we conduct WGPLRT and NLRT to get the local inferences yhat. Finally, yhat is passed to SBLUE to predict the labels at the un-monitored locations. 

Tests are also conducted on WGPLRT, NLRT, SBLUE, respectively. See [test_WGPLRT.m][e02],[test_NLRT.m][e03],[test_SBLUE.m][e04]. For WGPLRT and NLRT, ROC curves are plotted.

WGPLRT performs extremely well on differentiating Normal/Normal, Normal/Gamma warpings. However, the performance deteriorates drastically when the warping functions are Gamma/Gamma, Gamma/Beta.

NLRT performs reasonably overall with speed even faster than WGPLRT sometimes.


| No. | Description                                     | Code       |
| --- | ----------------------------------------------- | ---------- | 
| 1   | Test on Synthetic Dataset                       | [src][e01] | 
| 2   | Test for WGPLRT                                 | [src][e02] | 
| 3   | Test for NLRT                                   | [src][e03] | 
| 4   | Test for SBLUE                                  | [src][e04] | 
 





## LRT
The LRT folder contains the implementaion of [WGPLRT][e05] and [NLRT][e06]. 


## SBLUE
The SBLUE folder contains the impmentation of [SBLUE][e07]. 


[e01]: Experiment/test_Synthetic_Data.m
[e02]: Experiment/test_WGPLRT.m
[e03]: Experiment/test_NLRT.m
[e04]: Experiment/test_SBLUE.m
[e05]:LRT/WGPLRT/WGPLRT.m
[e06]:LRT/NLRT/NLRT.m
[e07]:SBLUE/SBLUE.m
