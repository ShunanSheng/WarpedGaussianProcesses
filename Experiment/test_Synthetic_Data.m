clear all,close all,clc
% Create a synthetic dataset and evaluate the performance on the dataset
% Create the synthetic dataset
% Setup for Spatial Random field
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp.cov=log([ell; sf]); 
% q=0.5; pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
c=0;pd=[];
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd,'thres',c);


%%% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 =1/2; sf0 = 1; hyp0.cov=log([ell0; sf0]);
pd0=makedist('Normal','mu',2,'sigma',4)
% pd0=makedist('Normal','mu',0,'sigma',1);
% pd0=makedist('Gamma','a',2,'b',4);

%%% H1 Alternative hypothesis

meanfunc1 = @meanConst; 
covfunc1 = {@covSEiso}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% covfunc1 = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% pd1=makedist('Gamma','a',1,'b',1)
% pd1=makedist('Beta','a',1,'b',1)
% pd1=makedist('Normal','mu',0,'sigma',1)
pd1=makedist('Logistic','mu',2,'sigma',5)

%%% Parameters for the sensor network
T=10; M=20; K=20; snP=0.1; snI=0.1;
modelHyp=struct("T",T,"M",M,"K",K,"snI",snI,"snP",snP);

%%% Lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
warpdist0="Normal";warpdist1="Normal";

[lb0,ub0]=lowUpBound(warpdist0,M);
[lb1,ub1]=lowUpBound(warpdist1,M);


SP=struct("meanfunc",meanfunc,"covfunc",covfunc,"hyp",hyp);
hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);


%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);
warpfunc_sf=@(c,x) indicator(c,x);

%%% Generate synthetic data
Data=SimSynData(SP,H0,H1,warpfunc_sf, warpfunc, modelHyp);

%%
clc;
x=Data.x;y=Data.y;indexTrain=Data.indexTrain;indexTest=Data.indexTest;
Xtrain=x(indexTrain,:);
Xtest=x(indexTest,:);

Ytrain=y(indexTrain);
Ytest=y(indexTest);

% The vector to store the decisions from LRT
Yhat=zeros(length(y),1);

% WGPLRT
t=Data.t;ZP0=Data.ZP.H0;ZP1=Data.ZP.H1;xP0=Data.xP.H0;xP1=Data.xP.H1;

% parameters
CP0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
muP0 = meanfunc0( hyp0.mean, t);
CP1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
muP1 = meanfunc1( hyp1.mean, t);

% run Laplace approximation
x_init=[ones(M,1)*0.5, ones(M,1)*0.5]; 
LRT=WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);

% find the logGamma at significance level alpha
alpha=0.05;
logGammaP=WGPLRT_opt_gamma(LRT,hyp0,CP0,muP0,warpfunc,t,snP,alpha);

% LRT for ZP0 and ZP1
yhat_pt_0=WGPLRT_pred(ZP0,LRT,logGammaP); % the classification
yhat_pt_1=WGPLRT_pred(ZP1,LRT,logGammaP);% the classification

% assign predictions to the locations
Yhat(xP0)=yhat_pt_0;
Yhat(xP1)=yhat_pt_1;

%% NLRT
% The Integral Observations
Z0=Data.ZI.H0;
Z1=Data.ZI.H1;
xI0=Data.xI.H0;
xI1=Data.xI.H1;

kw= ceil(exp(log(10000*T/K/180)/4)); % calculate the number of point neeed per window under Simpson's rule with 0.01 error
kw= round(kw/2)*2;n=kw*K;tI=linspace(0,T,n)';

CI0 = chol(feval(covfunc0{:}, hyp0.cov, tI)+1e-9*eye(n));
muI0 = meanfunc0( hyp0.mean, tI);

CI1 = chol(feval(covfunc1{:}, hyp1.cov, tI)+1e-9*eye(n));
muI1 = meanfunc1( hyp1.mean, tI);

sumstats=@summaryMoment; % the summary statistic
d=@distEuclid; % distance metric
J=100000; % number of samples per hypothesis
[ZI0,ZI1]=NLRT_gene(hyp0,CI0,muI0,hyp1,CI1,muI1, warpfunc,K,kw,snI,J); % generate J samples of integral observations from null ...                                                                   % and alternative hypothesis

% parameters for NLRT
delta=1; % distance tolerance
% logGammaI=NLRT_opt_logGamma(hyp0,CI0,muI0,ZI0,ZI1,warpfunc,sumstats,d,K,kw,snI,delta,alpha)
% In practice, NLRT performs so unstble that Lambda is most likely to be
% infinity, so we may set logGammaI=1 for simplicity.
logGammaI=log(1);

% NLRT for Z1
[D0,D1]=NLRT_stats(Z0,ZI0,ZI1,sumstats,d); % compute the distance matrix
Lambda0=NLRT_pred_delta(D0,D1,delta);
yhat_int_0=NLRT_pred_gamma(Lambda0,logGammaI); 

% NLRT for Z0
[D0,D1]=NLRT_stats(Z1,ZI0,ZI1,sumstats,d); % compute the distance matrix
Lambda1=NLRT_pred_delta(D0,D1,delta);
yhat_int_1=NLRT_pred_gamma(Lambda1,logGammaI); 

Yhat(xI0)=yhat_int_0;
Yhat(xI1)=yhat_int_1;

Ytrain_hat=Yhat(indexTrain);
%% Test performance of LRTs
% Overall
clc
[tp,fp]=confusionMat(Ytrain,Ytrain_hat);

display("Overall  "+":TPR="+tp+",FPR="+fp+",MSE="+sum((Ytrain-Ytrain_hat).^2))

% WGPLRT
YP_hat=[yhat_pt_0;yhat_pt_1];
YP=[y(xP0);y(xP1)];
[wtp,wfp]=confusionMat(YP,YP_hat);
display("WGPLRT  "+":TPR="+wtp+",FPR="+wfp+",MSE="+sum((YP-YP_hat).^2))

% NLRT
YI_hat=[yhat_int_0;yhat_int_1];
YI=[y(xI0);y(xI1)];
[ntp,nfp]=confusionMat(YI,YI_hat);

display("NLRT  "+":TPR="+ntp+",FPR="+nfp+",MSE="+sum((YI-YI_hat).^2))


%% SBLUE
clc;
% Offline phase, super super slow
tic
SBLUEprep=SBLUE_stats_prep(covfunc,meanfunc,hyp,Xtrain,Xtest); 
toc

%%
% Online phase: given the knowlegde of the LRT performance

liP=ismember(indexTrain,[xP0;xP1]);
liI=ismember(indexTrain,[xI0;xI1]);
rho=[1-wfp,1-nfp];lambda=[wtp,ntp];
A1=[rho(1),1-rho(1);1-lambda(1),lambda(1)];
A2=[rho(2),1-rho(2);1-lambda(2),lambda(2)];

transitionMat=SBLUE_confusion(A1,A2,liP,liI);
SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c);
Ypred=SBLUE_pred(SBLUE,Ytrain_hat);
[tp,fp]=confusionMat(Ytest,Ypred);
display("SBLUE  "+":TPR="+tp+",FPR="+fp+",MSE="+sum((Ytest-Ypred).^2/length(Ytest)))


