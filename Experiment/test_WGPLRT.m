%%% Test for WGPLRT on the simulated data
%%% WGPLRT works evey when testing the normal/gamma or normal/normal
%%% the performance is much worse when we are going to test gamma/gamma or
%%% gamma/beta


clear all,close all,clc

%%% Setup for Temproal processes

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
pd1=makedist('Gamma','a',5,'b',10);
% pd1=makedist('Beta','a',1,'b',1);
% pd1=makedist('Normal','mu',0,'sigma',1);

%%% Parameters for the sensor network
T=10; M=20; K=20; snP=0.1; snI=0.1; 

%%% lower/upper bound for optimization, the range of W

% lb0=[];ub0=[];lb1=[];ub1=[];  % normal/normal
lb0=zeros(M,1);ub0=[];lb1=zeros(M,1);ub1=[]; %gamma/gamma
% lb0=zeros(M,1);ub0=[];lb1=[];ub1=[];  % gamma/normal

hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);

%% test on WGPLRT (draw ROC)
% LogGamma=log([1]);
LogGamma=log([0.01,0.1,0.5,0.7,0.9,1,5,10,100])';
N=size(LogGamma,1);TP=zeros(N,1);FP=zeros(N,1);
for j=1:N
	n=5;yn=rand(n,1)>0.5; % Ground truth
    t=linspace(0,hyp0.t,M)';yhat=2*ones(n,1);logGamma=LogGamma(j);
    for i=1:n
        zP=SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,yn(i));
        yhat(i)=WGPLRT(zP,H0,H1,warpinv,t,snP,logGamma);
    end
    [tp,fp]=confusionMat(yn,yhat);
    TP(j)=tp;
    FP(j)=fp;
    display("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));
end

plotROC(TP,FP)
    
%% test on WGPLRT (single trial)
n=10;yn=rand(n,1)>0.5; % Ground truth

t=linspace(0,hyp0.t,M)';logGamma=log(exp(1));yhat=2*ones(n,1);

for i=1:n
    zP=SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,yn(i));
    yhat(i)=WGPLRT(zP,H0,H1,warpinv,t,snP,logGamma);
end
diff=yn-yhat;
figure()
scatter(1:n,diff)
sum(diff.^2)/n
sum(diff==0)/n

