%%% Test for WGPLRT on the simulated data
%%% WGPLRT works very well when testing the normal/gamma or normal/normal
%%% the performance is much worse when we are going to test gamma/gamma or
%%% gamma/beta


clear all,close all,clc

%%% Initialize Temproal processes

%%% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 =1/2; sf0 = 1; hyp0.cov=log([ell0; sf0]);
pd0=makedist('Normal','mu',10,'sigma',10)
% pd0=makedist('Normal','mu',2,'sigma',1)
% pd0=makedist('Gamma','a',2,'b',4)
% pd0=makedist('Logistic','mu',8,'sigma',2)
% pd0 = makedist('Stable','alpha',0.5,'beta',0,'gam',1,'delta',0)
% pd0=makedist('tLocationScale','mu',-2,'sigma',1,'nu',20)



%%% H1 Alternative hypothesis
meanfunc1 = @meanConst; 
covfunc1 = {@covSEiso}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% covfunc1 = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% pd1=makedist('Gamma','a',5,'b',10)
% pd1=makedist('Beta','a',1,'b',1)
% pd1=makedist('Logistic','mu',10,'sigma',10)
pd1=makedist('Normal','mu',0,'sigma',1)
% pd1=makedist('tLocationScale','mu',-2,'sigma',1,'nu',20)


%%% Parameters for the sensor network
T=50; M=100; snP=0.1;

%%% Lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
lb0=[];ub0=[];lb1=[];ub1=[];  % normal/normal, or any distribution with full support
% lb0=zeros(M,1);ub0=[];lb1=zeros(M,1);ub1=[]; %gamma/gamma
% lb0=zeros(M,1);ub0=[];lb1=[];ub1=[];  % gamma/normal
% lb0=[];ub0=[];lb1=zeros(M,1);ub1=[];  % normal/gamma


hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);

%% WGPLRT

clc;
n=100; % the size of 1d spatial field
t=linspace(0,hyp0.t,M)'; % the time points
yn=rand(n,1)>0.5; % the value of latent field

%% Test on WGPLRT (single trial)
clc;
yhat=2*ones(n,1); % initialize the decision vector
logGamma=0; % LRT threshold

% Parameters
C0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
mu0 = meanfunc0( hyp0.mean, t);
C1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
mu1 = meanfunc1( hyp1.mean, t);

% Run Laplace approximation
x_init=[ones(M,1)*1, ones(M,1)*1]; 
[LA0,LA1]=WGPLRT_opt(H0,H1,warpinv,t,x_init);

% Classification
for i=1:n
    zP=SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,yn(i));
    yhat(i)=WGPLRT_pred(zP,LA0,LA1,snP,logGamma);
end
diff=yn-yhat;
accuracy=(1-sum(diff.^2)/n)*100
[tp,fp]=confusionMat(yn,yhat)

%% Test on WGPLRT (draw ROC)
close all;clc;
LogGamma=log(linspace(0.01,20000,1000))';
N=size(LogGamma,1);
TP=zeros(N,1);FP=zeros(N,1);

% Parameters
C0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
mu0 = meanfunc0( hyp0.mean, t);
C1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
mu1 = meanfunc1( hyp1.mean, t);

% Run Laplace approximation
x_init=[ones(M,1)*0, ones(M,1)*0]; 
[LA0,LA1]=WGPLRT_opt(H0,H1,warpinv,t,x_init);

for j=1:N
	yn=rand(n,1)>0.5; % Ground truth
    logGamma=LogGamma(j); % The threshold
    for i=1:n
        zP=SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,yn(i));
        yhat(i)=WGPLRT_pred(zP,LA0,LA1,snP,logGamma);
    end
    % Compute the false/true positive rate
    [tp,fp]=confusionMat(yn,yhat);
    TP(j)=tp;
    FP(j)=fp;
    if mod(j,100)==0
        display("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));  
    end
end


%%
plotVector(FP)
plotVector(TP)
plotROC(TP,FP)

%% Locating the LRT threshold
alpha=0.05; % control the significance level to be 0.05
lgamma=0.01; % lower bound of the interval
rgamma=10; % upper bound of the interval
% Unable to plot meaning ROC (most likely to be the vertical line) or locating the threshold as the TP is either
% 0 or 1 in most cases
























