function ZP=SimFastPtData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,t,snP,n0,n1)
    % Simulate the point observations given the covariance matrix
    % and mean vector
    %
    % Input:
    % hyp0,hyp1: parameters for the two hypotheses
    % C0,C1    : chol decomposition of covariance matrix for H0,H1
    % mu0,mu1  : mean vector for H0,H1
    % n0,n1    : the number of point-observation sequences generated for
    % H0,H1
    % warpfunc : the warp function handle
    % t   : the time points to observe data
    % snP: noises for integral observations
    % Output: 
    % z : the point observations
    
    
    ZP0=SimPtData(hyp0,C0,mu0,warpfunc,t,snP,n0);
    ZP1=SimPtData(hyp1,C1,mu1,warpfunc,t,snP,n1);
    
    ZP=[ZP0,ZP1];
    
end
