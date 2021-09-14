function optLogGamma=WGPLRT_opt_gamma(LRT,hyp0,C0,mu0,warpfunc,t,snP,alpha)
    % Get the LRT threshold logGamma given the significance level alpha
    n=100000;
    ZP0=SimPtData(hyp0,C0,mu0,warpfunc,t,snP,n); % generate point observations from H0
    nlogLambda=testStats(ZP0,LRT); % compute the test statistics
    optLogGamma=-quantile(nlogLambda,1-alpha); % find optimal logGamma

end