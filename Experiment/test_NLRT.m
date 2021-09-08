%%% Test for NLRT on the simulated data

clear all,close all,clc

%%% Setup for Temporal processes
%%% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 =1/2; sf0 = 1; hyp0.cov=log([ell0; sf0]);
pd0=makedist('Normal','mu',10,'sigma',10);
% pd0=makedist('Normal','mu',0,'sigma',1);
% pd0=makedist('Gamma','a',2,'b',4);

%%% H1 Alternative hypothesis

meanfunc1 = @meanConst; 
covfunc1 = {@covSEiso}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% covfunc1 = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% pd1=makedist('Gamma','a',1,'b',1);
% pd1=makedist('Beta','a',1,'b',1);
pd1=makedist('Normal','mu',0,'sigma',1);

%%% Parameters for the sensor network
T=10; K=100; snI=0.1; 



% lb, ub is only used for WGPLRT, however for completeness of the
% initialization process, we inclide lb/ub here
warpdist0="Normal";warpdist1="Normal";M=50;

[lb0,ub0]=lowUpBound(warpdist0,M);
[lb1,ub1]=lowUpBound(warpdist1,M);

hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);

%% NLRT
clc;
nI=10000;n0=nI*0.5;n1=nI-n0;
yn=[zeros(n0,1);ones(n1,1)]; % ground truth, the value of latent field, 
% half the null hypothesis and half the alternative hypothesis

kw= ceil(exp(log(10000*T/K/180)/4)); % calculate the number of point neeed per window under Simpson's rule with 0.01 error
kw= round(kw/2)*2;n=kw*K;x=linspace(0,T,n)';
    
C0 = chol(feval(covfunc0{:}, hyp0.cov, x)+1e-9*eye(n));
mu0 = meanfunc0( hyp0.mean, x);

C1 = chol(feval(covfunc1{:}, hyp1.cov, x)+1e-9*eye(n));
mu1 = meanfunc1( hyp1.mean, x);

% the integral observations
ZI=SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,K,kw,snI,n0,n1);

%% NLRT ROC
clc;
sumstats=@summaryMoment; % the summary statistic
d=@distEuclid; % distance metric
delta=1; % distance tolerance
J=10000; % number of samples per hypothesis

[ZI0,ZI1]=NLRT_gene(hyp0,C0,mu0,hyp1,C1,mu1, warpfunc,K,kw,snI,J); % generate J samples of integral observations from null ...
                                                                   % and alternative hypothesis

%%
clc;
N=1000;LogGamma=linspace(-100,100,N);
TP=zeros(N,1);FP=zeros(N,1);
[D0,D1]=NLRT_stats(ZI,ZI0,ZI1,sumstats,d); % compute the distance matrix

tic;
for j=1:N
    tstart=tic;
    logGamma=LogGamma(j);
    [Lambda,yhat]=NLRT_pred(D0,D1,delta,logGamma); % Compute yhat given delta and logGamma
    [tp,fp]=confusionMat(yn,yhat);
    TP(j)=tp;
    FP(j)=fp;
    if mod(j,100)==0
        display("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));
        telapsed = toc(tstart)
    end
end
avergeTime=toc/N
plotROC(TP,FP)

%% find the optimal logGamma
logGamma=0;
J=10000; % number of samples per hypothesis


[ZI0,ZI1]=NLRT_gene(hyp0,C0,mu0,hyp1,C1,mu1, warpfunc,K,kw,snI,J);

ZInull=SimIntData(hyp0,C0,mu0, warpfunc,K,kw,snI,nI);

[D0,D1]=NLRT_stats(ZInull,ZI0,ZI1,sumstats,d); % compute the distance matrix

%%
clc
Delta=linspace(0.01,20,20);TP=zeros(20,1);FP=zeros(20,1);
for i=1:20
    [Lambda,yhat]=NLRT_pred(D0,D1,Delta(i),logGamma);
%     histogram(Lambda)
    alpha=0.05; % significance Level
    optGamma=quantile(Lambda,alpha); % find optimal logGamma
    [Lambda,yhat]=NLRT_pred(D0,D1,Delta(i),log(optGamma)); % Compute yhat given delta and logGamma
    [tp,fp]=confusionMat(yn,yhat);
    FP(i)=fp;
    TP(i)=tp;
end

plotROC(TP,FP)

%%
clc;
histogram(Lambda)
alpha=0.05 % significance Level
optGamma=quantile(Lambda,alpha) % find optimal logGamma
[Lambda,yhat]=NLRT_pred(D0,D1,delta,log(optGamma)); % Compute yhat given delta and logGamma
[tp,fp]=confusionMat(yn,yhat)



%%%

% 
% % Single Trial
% clc;
% gamma=1;
% yhat=NLRT_pred(ZI,S0,S1,sumstats,d,delta,gamma);
% [tp,fp]=confusionMat(yn,yhat);
% %%%