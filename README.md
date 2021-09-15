# Warped Gaussian Process Classfication

## Experiment
Multiple expriments have been conducted to validate our algorithms. In the [test_Synthetic_Data.m][e01], we create a synthetic dataset of sensor network depolyed in R^2 with 2500 spatial points over the simplex [-1,1] x [-1,1]. At each location, based on the value of the spatial field, a sequence of point/integral observastions is generated from either H0 or H1. 

For each sequence at monitored locations, Xtrian, we conduct WGPLRT and NLRT to make the local inference, i.e. to classifiy H0/H1. Finally, Yhat is passed to SBLUE to predict the labels at the un-monitored locations, Xtest. 

To test the efficacy of individual algorithms, we also conduct experiments on WGPLRT, NLRT, SBLUE respectively. See [test_WGPLRT.m][e02],[test_NLRT.m][e03],[test_SBLUE.m][e04]. ROC curves are plotted for all algorithms, see [fig][e08]. We find that
 - WGPLRT performs extremely well on differentiating distributions with full supports. However, the performance deteriorates drastically when the warping functions are Gamma/Gamma, Gamma/Beta, i.e. the distributions with partial supports. This is due to the failure of Laplaca Approximation to these distributions. Nontheless, WGPLRT may also be applicable when the distributions have densities concentrated within the interior of the support.

 - NLRT performs reasonably well over multiple cases. However, it may not be the best choice when the data size is large, e.g. N>100,000, becuase NLRT computes the distance D0, D1 between each observation and the samples generated and creates matrices of size NxJ, J is the size of samples generated.
 
 - SBLUE works quite well for the noisy data. However, when the confusion matrix is around 0.5 * eye(2). The performance becomes very poor due to the equivocal information from the data.


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
[e08]:fig
