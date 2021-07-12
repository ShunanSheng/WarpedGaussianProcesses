# Warped Gaussian Process Classfication

## Experiment
Multiple expriments have been conducted to validate our algorithm. In the [test_Synthetic_Data][e01], we create a synthetic dataset of sensor network depolyed in R2. Based on the data, we conduct WGPLRT and NLRT to get the local inferences yhat. Finally, yhat is passed to SBLUE to predict the labels at the un-monitored locations. 

Tests are also conducted on WGPLRT, NLRT, SBLUE, respectively. See [src][e02][src][e03][src][e04]


| No. | Description                                     | Code       |
| --- | ----------------------------------------------- | ---------- | 
| 1   | Test on Synthetic Dataset                       | [src][e01] | 
| 2   | Test for WGPLRT                                 | [src][e02] | 
| 3   | Test for NLRT                                   | [src][e03] | 
| 4   | Test for SBLUE                                  | [src][e04] | 
 


[e01]: Experiment/test_Synthetic_Data.m
[e02]: Experiment/test_WGPLRT.m
[e03]: Experiment/test_NLRT.m
[e04]: Experiment/test_SBLUE.m




