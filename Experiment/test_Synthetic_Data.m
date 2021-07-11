clear all,close all,clc
% Create a synthetic dataset and evaluate the performance on the dataset
% Create the synthetic dataset
% Setup for Spatial Random field
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 2; sf = 1; hyp.cov=log([ell; sf]); q=0.5;
pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);


%%% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 =1/2; sf0 = 1; hyp0.cov=log([ell0; sf0]);
% pd0=makedist('Normal','mu',3,'sigma',4);
% pd0=makedist('Normal','mu',0,'sigma',1);
pd0=makedist('Gamma','a',2,'b',4);

%%% H1 Alternative hypothesis

meanfunc1 = @meanConst; 
covfunc1 = {@covSEiso}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% covfunc1 = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
pd1=makedist('Gamma','a',1,'b',1);
% pd1=makedist('Beta','a',1,'b',1);
% pd1=makedist('Normal','mu',0,'sigma',1);

%%% Parameters for the sensor network
T=10; M=20; K=20; snP=0.1; snI=0.1;
modelHyp=struct("T",T,"M",M,"K",K,"snI",snI,"snP",snP);

%%% lower/upper bound for optimization, the range of W

% lb0=[];ub0=[];lb1=[];ub1=[];  % normal/normal
lb0=zeros(M,1);ub0=[];lb1=zeros(M,1);ub1=[]; %gamma/gamma
% lb0=zeros(M,1);ub0=[];lb1=[];ub1=[];  % gamma/normal

SP=struct("meanfunc",meanfunc,"covfunc",covfunc,"hyp",hyp);
hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);


%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);


%%% Generate synthetic data
[ZP,ZI,y,xP,xI,indexTrain,indexTest,x]=SimSynData(SP,H0,H1,warpfunc,modelHyp);
Xtrain=x(indexTrain,:);
xstar=x(indexTest,:);

Ytrue=y(indexTest);
Yhat=y;

NP=size(ZP,1);
NI=size(ZI,1);

%% Conduct LRTs
% Conduct WGPLRT on ZP 
logGamma=log(1);t=linspace(0,hyp0.t,M)';
for i=1:NP
    zP=ZP(i,:)';
    Yhat(xP(i))=WGPLRT(zP,H0,H1,warpinv,t,snP,logGamma);
end
display("done")

%%
% Conduct NLRT on ZI 
sumstats=@summaryMoment;
d=@distEuclid;
delta=1;
gamma=1;

for i=1:NI
    zI=ZI(i,:)';
    Yhat(xI(i))=NLRT(zI,H0,H1,warpfunc,K,snI,sumstats,d,delta,gamma);
end

display("done")

%% Conduct SBLUE
rho=0.8;
A=[rho,1-rho;1-rho,rho];

Ytrain=Yhat(indexTrain);
Ypred=SBLUE(covfunc,hyp.cov,Ytrain,Xtrain,xstar,A,q); % Predictions


Ydiff=(Ypred-Ytrue)';
MSE=sum(Ydiff.^2)/length(Ydiff)
Accuracy=sum(Ydiff==0)/length(Ydiff)





