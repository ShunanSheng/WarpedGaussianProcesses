function [TP, FP] = FuncWGPLRTroc(H0, H1, T, M, snP, printOpt,figOpt, LRT)
%%% Compute the TP, FP of NLRT to plot ROC given the parameters
%   Input:
%            H0 : null hypothesis
%            H1 : alternative hypothesis
%            T : the time frame, i.e., [0,T}
%            M : the number of point observations
%            snP : noise variance for point sensors
%            printOpt : logical variable controling printing status
%            figOpt : logical variable controlling plotting status
%            LRT : precalculated LRT value
%   Output:
%            TP - the list of true positive rates
%            FP - the list of false positive rates

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


%% WGPLRT
n = 10000; % the size of the spatial field
t = linspace(0,T,M)'; % the time points

% run Laplace approximation if LRT is not given
if ~exist('LRT', 'var') || isempty(LRT)
    x_init = [ones(M,1)*pd0.mean, ones(M,1)*pd1.mean]; 
    LRT = WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);
end
% parameters
C0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
mu0 = meanfunc0( hyp0.mean, t);
C1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
mu1 = meanfunc1( hyp1.mean, t);

% half the null hypothesis and half the alternative hypothesis
n0 = 0.5 * n;
n1 = n - n0;
yn = [zeros(n0,1);ones(n1,1)]; % ground truth, the value of latent field, 

% generate samples
ZP = SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,n0,n1);

%% Plot ROC
LogGamma = [linspace(-500, 500, 2000),linspace(500,6000, 500),linspace(6000,100000,100)];
N = length(LogGamma);
TP=zeros(N,1);FP=zeros(N,1);

for j=1:N
    logGamma = LogGamma(j); % the threshold
    yhat = WGPLRT_pred(ZP,LRT,logGamma); % the classification
    % compute the false/true positive rate
    [tp,fp] = confusionMat(yn,yhat);
    TP(j) = tp;
    FP(j) = fp;
    if printOpt == true
        if mod(j,200)==0
            disp("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));  
        end
    end
end

if figOpt == true
    plotROC(TP,FP)
end

end
