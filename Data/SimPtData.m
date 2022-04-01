function ZP=SimPtData(hyp,C,mu,warpfunc,t,snP,nP)
    % Simulate point observations at time points t given single hyp,
    % C,mu,warpfunc,t,snP (generate point observations for a single hypothesis)
    %
    % Input:
    % hyp: parameters of the given hypothesis
    % C   : chol decomposition of covariance matrix for H0/H1
    % mu  : mean vector for H0/H1
    % warpfunc : the warp function handle
    % t   : the time points to observe data
    % snP: noises for integral observations
    % nP  : the number of point-observation sequences  generated
    % Output: 
    % ZP : the point observations

    
    M=size(t,1);
    
    % Define the latent Gaussian Process
    f = C'*randn(M, nP) + mu;
    % Define the warped Gaussian Process, then the point observations
    ZP = warpfunc(hyp.dist,f)+snP*randn(M,nP);
end
