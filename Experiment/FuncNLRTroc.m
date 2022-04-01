function [TP, FP] = FuncNLRTroc(H0, H1, T, K, snI, printOpt, figOpt, ZI0, ZI1)
%%% Compute the TP, FP of NLRT to plot ROC given the parameters
%   Input: 
%            H0 : null hypothesis
%            H1 : alternative hypothesis
%            T : the time frame, i.e., [0,T}
%            K : the number of intergal observations
%            snI : noise variance for integral sensors
%            printOpt : logical variable controling printing status
%            figOpt : logical variable controlling plotting status
%   Output:
%           TP - the true positive rates
%           FP - the false positive rates

%% the null and alternative hypotheses
meanfunc0 = H0.meanfunc; 
covfunc0 = H0.covfunc;
hyp0 = H0.hyp;
pd0 = hyp0.dist;

meanfunc1 = H1.meanfunc; 
covfunc1 = H1.covfunc;
hyp1 = H1.hyp;
pd1 = hyp1.dist;

% warping function
warpfunc = @(pd,p) invCdf(pd,p);
warpinv = @(pd,p) invCdfWarp(pd,p);

%% NLRT
% ground truth, the value of latent field, 
% half the null hypothesis and half the alternative hypothesis
nI = 10000;
n0 = nI*0.5;
n1 = nI-n0;
yn = [zeros(n0,1);ones(n1,1)]; 

% calculate the number of point neeed per window under Simpson's rule with 0.01 error
kw = ceil(exp(log(1000000*(T/K).^5/180)/4)); 
kw = round(kw/2)*2;
if kw < 4
    kw = 4; % at least four points in an integration window
end
n = kw * K;
x = linspace(0,T,n)';
    
C0 = chol(feval(covfunc0{:}, hyp0.cov, x)+1e-9*eye(n));
mu0 = meanfunc0( hyp0.mean, x);

C1 = chol(feval(covfunc1{:}, hyp1.cov, x)+1e-9*eye(n));
mu1 = meanfunc1( hyp1.mean, x);

% the integral observations
ZI = SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,K,kw,snI,n0,n1);
%% NLRT ROC curve constants
sumstats=@(z) summaryAuto(z,4); % the summary statistic

d = @distEuclid; % distance metric

if ~exist('ZI0', 'var') || isempty(ZI0) || ~exist('ZI1', 'var') || isempty(ZI1)
    J=10000; % number of samples per hypothesis
    % generate J samples of integral observations from null and alternative hypotheses
    [ZI0,ZI1]=NLRT_gene(hyp0,C0,mu0,hyp1,C1,mu1, warpfunc,K,kw,snI,J); 
end

[D0,D1] = NLRT_stats(ZI,ZI0,ZI1,sumstats,d); % compute the distance matrix
%% Plot ROC
LogGamma = [linspace(-500, 500, 2000),linspace(500,6000, 500),linspace(6000,100000,100)];
N = length(LogGamma);
optDelta = 0.1;

TP = zeros(N,1);
FP = zeros(N,1);

Lambda=NLRT_pred_delta(D0,D1,optDelta); % test statistics

for j=1:N
    logGamma=LogGamma(j);
    % compute yhat given Lambda and logGamma
    yhat=NLRT_pred_gamma(Lambda,logGamma); 
    [tp,fp]=confusionMat(yn,yhat);
    TP(j)=tp;
    FP(j)=fp;
    if printOpt==true
        if mod(j,500)==0
            display("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));
        end
    end
end

if figOpt==true
    %% plot the ROC
    plotROC(TP,FP,"ROC:NLRT","delta="+optDelta)
end


end