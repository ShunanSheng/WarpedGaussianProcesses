function z=SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,yx)
    % Simulate the point observations given the covariance matrix
    % and mean vector
    %
    % Input:
    % hyp0,hyp1: parameters for the two hypotheses
    % C0,C1    : chol decomposition of covariance matrix for H0,H1
    % mu0,mu1  : mean vector for H0,H1
    % warpfunc : the warp function handle
    % K  : total numebr of windows over [0,T]
    % kw : the number of points per window
    % snI: noises for integral observations
    % yx : binary value deciding H0 or H1
    % Output: 
    % zI : the integral observations

    if yx==0
        C=C0;
        mu=mu0;
        hyp=hyp0;
    else
        C=C1;
        mu=mu1;
        hyp=hyp1;
    end
    
    T=hyp.t;
    
    M=size(t,1);
    
    % Define the latent Gaussian Process
    f = C'*randn(M, 1) + mu;
    % Define the warped Gaussian Process, then the point observations
    z=warpfunc(hyp.dist,f)+snP*randn(M,1);
    
end
