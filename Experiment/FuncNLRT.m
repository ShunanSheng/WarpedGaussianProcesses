function [tp,fp,optLogGamma] = FuncNLRT(H0, H1, T, K, snI, alpha, printOpt, figOpt,Z0, Z1)
% Given the input of signal variance and number of windows, output the
% corresponding tp, fp and optimal threshold



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
n = kw*K;
x = linspace(0,T,n)';
    
C0 = chol(feval(covfunc0{:}, hyp0.cov, x)+1e-9*eye(n));
mu0 = meanfunc0( hyp0.mean, x);

C1 = chol(feval(covfunc1{:}, hyp1.cov, x)+1e-9*eye(n));
mu1 = meanfunc1( hyp1.mean, x);

% the integral observations
ZI = SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,K,kw,snI,n0,n1);
%% NLRT ROC curve constants
sumstats = @(z) summaryAuto(z,4); % the summary statistic

d = @distEuclid; % distance metric

if ~exist('Z0', 'var') || isempty(Z0) || ~exist('Z1', 'var') || isempty(Z1)
    J = 10000; % number of samples per hypothesis
    % generate J samples of integral observations from null and alternative hypotheses
    [Z0,Z1] = NLRT_gene(hyp0,C0,mu0,hyp1,C1,mu1,warpfunc,K,kw,snI,J); 
end

[D0,D1] = NLRT_stats(ZI,Z0,Z1,sumstats,d); % compute the distance matrix
       
%% find the optimal logGamma
delta = 0.1;
optLogGamma = NLRT_opt_logGamma(hyp0,C0,mu0,Z0,Z1,warpfunc,sumstats,d,K,kw,snI,delta,alpha);

%% The performance at the opt_logGamma
Lambda = NLRT_pred_delta(D0,D1,delta);
yhat = NLRT_pred_gamma(Lambda,optLogGamma); % Compute yhat given delta and logGamma
[tp,fp] = confusionMat(yn,yhat);


%% Plot ROC
if figOpt
    N = 1000;
    M = 10; 
    LogGamma = linspace(-1000,1000,N);
    Delta = linspace(0,1,M);% distance tolerance
    TP = zeros(N,M);
    FP = zeros(N,M);

    for i = 1:M
        delta = Delta(i);
        Lambda = NLRT_pred_delta(D0,D1,delta);
        for j=1:N
            logGamma = LogGamma(j);
            yhat = NLRT_pred_gamma(Lambda,logGamma); % Compute yhat given Lambda and logGamma
            [tp,fp] = confusionMat(yn,yhat);
            TP(j,i) = tp;
            FP(j,i) = fp;
            if printOpt==true
                if mod(j,500)==0
                    display("Iteration="+j+",TP="+TP(j,i)+",FP="+FP(j,i));
                end
            end
        end
    end

    % Plot the ROC graph
    FigLegend=cell(M,1);
    for i=1:M
       FigLegend{i}="delta="+Delta(i); % create legend recording delta
    end

    plotROC(TP,FP,[],FigLegend)
    %% plot ROC given the optimal delta
    TP=zeros(N,1);FP=zeros(N,1);
    optDelta=0.1;
    Lambda=NLRT_pred_delta(D0,D1,optDelta);
    for j=1:N
        logGamma = LogGamma(j);
        yhat = NLRT_pred_gamma(Lambda,logGamma); % Compute yhat given Lambda and logGamma
        [tp,fp] = confusionMat(yn,yhat);
        TP(j) = tp;
        FP(j) = fp;
        if printOpt
            if mod(j,500)==0
                display("Iteration="+j+",TP="+TP(j)+",FP="+FP(j));
            end
        end
    end
    plotROC(TP,FP,"ROC:NLRT","delta="+optDelta)
end

end
