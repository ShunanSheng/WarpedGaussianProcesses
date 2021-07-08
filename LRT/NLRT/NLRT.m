function yhat=NLRT(zI,H0,H1,warpfunc,K,snI,sumstats,d,delta,gamma)
    % Conduct Neighbourhood density based LRT given the integral observations 
    %
    % Input: 
    % zI   : the integral observations
    % H0,H1: parameters for null/alternative hypotheses
    % K    : the number of windows
    % snI  : noise for integral observations
    % sumstats : summary statistics 
    % d    : distance mertic 
    % delta: distance tolerance
    % gamma: LRT thereshold
    %
    % Output: 
    % yhat : the decision
    
    meanfunc0 = H0.meanfunc;
    covfunc0 = {H0.covfunc};
    hyp0=H0.hyp;

    meanfunc1 = H1.meanfunc; 
    covfunc1 = {H1.covfunc};
    hyp1=H1.hyp;
    
    T=hyp1.t;
    
    kw=100;n=kw*K;x=linspace(0,T,n)';
    
    C0 = chol(feval(covfunc0{:}, hyp0.cov, x)+1e-9*eye(n));
    mu0 = meanfunc0( hyp0.mean, x);

    C1 = chol(feval(covfunc1{:}, hyp1.cov, x)++1e-9*eye(n));
    mu1 = meanfunc1( hyp1.mean, x);
    
    
    J=20;n0=1;n1=1;y=sumstats(zI);
   % Generating data
    for j=1:J
        z0=SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,kw,K,snI,0);
        z1=SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,kw,K,snI,1);
%         display("distance 0="+d(sumstats(z0),y))
%         display("distance 1="+d(sumstats(z1),y))
          % Reject sample
        if d(sumstats(z0),y)<delta
            n0=n0+1;
        end
        if d(sumstats(z1),y)<delta
            n1=n1+1;
        end
    end
    display("n0="+n0+",n1="+n1)
    
    Lambda=n0/n1;
    yhat=Lambda<gamma;
   
end
