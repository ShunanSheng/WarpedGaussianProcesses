% pd = makedist("Beta", "a", 2, "b", 5);
% pd = makedist("Gamma", "a", 2, "b", 5);
% pd = makedist("Exponential", 2);
% pd = makedist("tLocationScale", "mu", 1, "sigma", 1, "nu",5);
pd = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);
warpfunc = @(pd,p) invCdf(pd,p);
warpinv = @(pd,p) invCdfWarp(pd,p);
W = @(x) warpfunc(pd,x);
G = @(v) warpinv(pd,v);
N = 10000;
x = sort(normrnd(0, 1, [N, 1]), 'ascend');
v = W(x);
dG = gradientG(pd,G,v);

plot(x, log(dG))

% histogram(v)

%%
pd = makedist('Normal');
t = truncate(pd,-2,Inf);
x = linspace(-3,10,1000);
figure
plot(x,pdf(pd,x))
hold on
plot(x,pdf(t,x),'LineStyle','--')
legend('Normal','Truncated')
hold off

%%
histogram(random(t, 100000, 1))

%% 
meanfunc0 = @meanConst; 
covfunc0 = {@covMaterniso, 1}; ell0=1; sf0=1; hyp0.cov=log([ell0; sf0]);
pd0 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1)


%%% H1 Alternative hypothesis

meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 5}; ell1=1; sf1=1; hyp1.cov=log([ell1; sf1]);
pd1 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1)


%%% Parameters for the sensor network
T=20; K= 20; snI= 0.1; 

% lb, ub is only used for WGPLRT, however for completeness of the
% initialization process, we inclide lb/ub here
warpdist0="Normal";warpdist1="Normal";M=50;

[lb0,ub0]=lowUpBound(warpdist0,M);
[lb1,ub1]=lowUpBound(warpdist1,M);

hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",{covfunc0},"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",{covfunc1},"hyp",hyp1);

%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);

%% NLRT
tic
clc;
nI=1000;n0=nI*0.5;n1=nI-n0;
yn=[zeros(n0,1);ones(n1,1)]; % ground truth, the value of latent field, 
% half the null hypothesis and half the alternative hypothesis

kw= ceil(exp(log(10000*T/K/180)/4)); % calculate the number of point needed per window under Simpson's rule with 0.01 error
kw= round(kw/2)*2;n=kw*K;x=linspace(0,T,n)';
    
C0 = chol(feval(covfunc0{:}, hyp0.cov, x)+1e-9*eye(n));
mu0 = meanfunc0( hyp0.mean, x);

C1 = chol(feval(covfunc1{:}, hyp1.cov, x)+1e-9*eye(n));
mu1 = meanfunc1( hyp1.mean, x);

% the integral observations
ZI = SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,K,kw,snI,n0,n1);

%% 
m = mean(ZI, 2)

