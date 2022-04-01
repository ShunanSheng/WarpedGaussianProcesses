function [tp,fp,optLogGamma]=FuncSimWGPLRT(M,sn,alpha,printOpt,figOpt)
% Given the input of signal variance and number of points, output the
% corresponding tp, fp and optimal threshold

%%% Initialize Temporal processes
%%% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covMaterniso, 1}; ell0=1; sf0=1; hyp0.cov=log([ell0; sf0]);
pd0 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);


%%% H1 Alternative hypothesis
meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 5}; ell1=1; sf1=1; hyp1.cov=log([ell1; sf1]);
pd1 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);


%%% Parameters for the sensor network
T=20; snP=sn; 
% each point observation zP is of size Mx1 with noise ~ N(0,snP^2I)

%%% Lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
warpdist0="Normal";warpdist1="Normal";
% warpdist0="Gamma";warpdist1="Gamma";
% warpdist0="Normal";warpdist1="Gamma";

[lb0,ub0]=lowUpBound(warpdist0,M);
[lb1,ub1]=lowUpBound(warpdist1,M);


% For distribution without full support, we require the density around
% boundary to be near zero;
% Approximation to be accurate when the tail probability decays fast enough

hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);

%% WGPLRT

n=10000; % the size of 1d spatial field
t=linspace(0,hyp0.t,M)'; % the time points

% half the null hypothesis and half the alternative hypothesis
n0=0.5*n;n1=n-n0;
yn=[zeros(n0,1);ones(n1,1)]; % ground truth, the value of latent field, 

% parameters
C0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
mu0 = meanfunc0( hyp0.mean, t);
C1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
mu1 = meanfunc1( hyp1.mean, t);

% run Laplace approximation
x_init=[ones(M,1)*pd0.mean+3, ones(M,1)*pd1.mean]; 
LRT=WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);

% generate samples
ZP=SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,n0,n1);

%% Plot ROC
if figOpt==true
    N=2000;LogGamma=linspace(-500, 500,N)';
    TP=zeros(N,1);FP=zeros(N,1);

    for j=1:N
        logGamma=LogGamma(j); % the threshold
        yhat=WGPLRT_pred(ZP,LRT,logGamma); % the classification
        % compute the false/true positive rate
        [tp,fp]=confusionMat(yn,yhat);
        TP(j)=tp;
        FP(j)=fp;
        if printOpt == true
            if mod(j,100)==0
                disp("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));  
            end
        end
    end
    plotROC(TP,FP)
end

%% Locating the LRT threshold

optLogGamma=WGPLRT_opt_gamma(LRT,hyp0,C0,mu0,warpfunc,t,snP,alpha);
logGamma=optLogGamma; % compute for n values with nhat observations in one batch

yhat=WGPLRT_pred(ZP,LRT,logGamma);
[tp,fp]=confusionMat(yn,yhat);
if printOpt==true
    disp("At significance level="+alpha+", the optlogGamma="+logGamma+", tp="+tp+",fp="+fp);
end





















end