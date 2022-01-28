function [tp,fp,optLogGamma]=FuncWGPLRT(H0, H1, T, M, snP,alpha, printOpt,figOpt, LRT)
% Given the null and alternative hypotheses, output tpr, fpr and the
% optimal LRT threshold provided the significance level alpha
meanfunc0 = H0.meanfunc; 
covfunc0 = H0.covfunc;
hyp0=H0.hyp;
pd0=hyp0.dist;

meanfunc1 = H1.meanfunc; 
covfunc1 = H1.covfunc;
hyp1=H1.hyp;
pd1=hyp1.dist;

%%% warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);

%% WGPLRT

n = 10000; % the size of 1d spatial field
t = linspace(0,T,M)'; % the time points

% run Laplace approximation if LRT is not given
if ~exist('LRT', 'var') || isempty(LRT)
    x_init = [ones(M,1)*pd0.mean+3, ones(M,1)*pd1.mean]; 
    LRT = WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);
end
% parameters
C0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
mu0 = meanfunc0( hyp0.mean, t);
C1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
mu1 = meanfunc1( hyp1.mean, t);

% half the null hypothesis and half the alternative hypothesis
n0 = 0.5*n;
n1 = n-n0;
yn = [zeros(n0,1);ones(n1,1)]; % ground truth, the value of latent field, 

% generate samples
ZP = SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,n0,n1);

%% Plot ROC
if figOpt == true
    N = 2000;LogGamma = linspace(-500, 500,N)';
    TP = zeros(N,1);FP = zeros(N,1);
    for j=1:N
        logGamma = LogGamma(j); % the threshold
        yhat = WGPLRT_pred(ZP,LRT,logGamma); % the classification
        % compute the false/true positive rate
        [tp,fp] = confusionMat(yn,yhat);
        TP(j) = tp;
        FP(j) = fp;
        if printOpt == true
            if mod(j,100) == 0
                disp("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));  
            end
        end
    end
    plotROC(TP,FP)
end

%% Locating the LRT threshold

optLogGamma = WGPLRT_opt_gamma(LRT,hyp0,C0,mu0,warpfunc,t,snP,alpha);
logGamma = optLogGamma; % compute for n values with nhat observations in one batch

yhat = WGPLRT_pred(ZP,LRT,logGamma);
[tp,fp] = confusionMat(yn,yhat);
if printOpt == true
disp("At significance level="+alpha+", the optlogGamma="+logGamma+", tp="+tp+",fp="+fp);
end






end