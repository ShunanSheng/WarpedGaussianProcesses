# Binary Spatial Random Field Reconstruction from Non-Gaussian Inhomogeneous Time-series Observations
[update on 06 April 2022]: include synthetic and real-world (semisynthetic) experiments; clean up codes.

[update on 06 Oct 2021]: include the g_and_h distribution into makedist, allowing SBLUE to take different transition matrix at each sensor.

## Paper
Put on the ArXiv Link:


## Dataset 
The weather dataset from NEA Singapore for the real-world experiments was retrived from https://data.gov.sg/search?groups=environment on 7 Jan 2022.

The Singapore boundary file (.shp) was retrieved from https://maps.princeton.edu/catalog/stanford-pg798kr1205 on 20 Jan 2022.


## Experiment
Multiple expriments have been conducted to validate our algorithms. In the [synthetic_experiment.m][e01], we create a synthetic dataset of sensor network depolyed in R^2 with 2500 spatial locations over the simplex [-5,5] x [-5,5]. At each sensor location, based on the value of the spatial field, a sequence of point/integral observations is generated from either H0 or H1. 

Based on the sensor type, each sensor perfroms WGPLRT or NLRT to make the local inference, i.e. to classify H0/H1. Then, SBLUE is used to reconstruct the binary spatial field at locations where no sensors are placed.

In the [semi_synthetic_experiment.m][e02], the experiments are performed based on a real dataset from NEA Singapore.

To test the efficacy of individual algorithms, we also conduct experiments on WGPLRT, NLRT, SBLUE respectively. See [test_WGPLRT.m][e02],[test_NLRT.m][e03],[test_SBLUE.m][e04]. 

## Results
Please refer to the paper for the detailed results.

| No. | Description                                     | Code       |
| --- | ----------------------------------------------- | ---------- | 
| 1   | Synthetic Experiments                           | [src][e01] |
| 1   | Real-world Experiments                          | [src][e02] | 
| 2   | Test for WGPLRT                                 | [src][e03] | 
| 3   | Test for NLRT                                   | [src][e04] | 
| 4   | Test for SBLUE                                  | [src][e05] | 
 

[e01]: Experiment/SyntheticExperiment/synthetic_experiment.m
[e02]: Experiment/SemiSyntheticExperiment/semi_synthetic_experiment.m
[e03]: Experiment/test_WGPLRT.m
[e04]: Experiment/test_NLRT.m
[e05]: Experiment/test_SBLUE.m

## Disclaimer
During the implementation of the codes, we use the GPML package provided by Carl Edward Rasmussen & Hannes Nickisch, which is accessible on http://gaussianprocess.org/gpml/code/matlab/doc/.
