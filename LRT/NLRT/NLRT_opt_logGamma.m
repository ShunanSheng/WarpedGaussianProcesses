function optlogGamma=NLRT_opt_logGamma(hyp0,C0,mu0,ZI0,ZI1,warpfunc,sumstats,d,K,kw,snI,delta,alpha)
    % Compute the logGamma given certain significance level
    
   
    nI = 10000; % number of samples per hypothesis
    ZInull = SimIntData(hyp0,C0,mu0, warpfunc,K,kw,snI,nI);% generate integral observations from H0 only
    [D0,D1] = NLRT_stats(ZInull,ZI0,ZI1,sumstats,d); % compute the distance matrix
    Lambda = NLRT_pred_delta(D0,D1,delta);% the optimal delta from ROC
    optlogGamma = log(quantile(Lambda,alpha)); % find optimal logGamma

end