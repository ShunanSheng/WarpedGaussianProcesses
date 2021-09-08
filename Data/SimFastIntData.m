function ZI=SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,K,kw,snI,n0,n1)
    % Simulate the integral observations given the covariance matrix
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
    % n0,n1: the number of integral observation sequences for H0,H1
    % Output: 
    % ZI : the integral observations

    ZI0=SimIntData(hyp0,C0,mu0, warpfunc,K,kw,snI,n0);
    ZI1=SimIntData(hyp1,C1,mu1, warpfunc,K,kw,snI,n1);
    
    ZI=[ZI0,ZI1];
    
 
end