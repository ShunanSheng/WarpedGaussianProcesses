function ZI=SimIntData(hyp,C,mu, warpfunc,K,kw,snI,nI)
    % Simulate integral observations over time period [0,T] with K windows
    % using the given chol matrix C, and mean vector C
    %
    % Input: 
    % hyp:  parameters for the two hypotheses
    % C  : chol decomposition of covariance matrix
    % mu : mean vector
    % warpfunc: the warp function handle
    % K  : the total numebr of windows over [0,T]
    % kw : the number of points per window
    % snI: noises for integral observations
    % nI  : the number of integral-observation sequences  generated
    %
    % Output: 
    % ZI : the integral observations

    % parameters
    T = hyp.t;
    n = kw*K;
    dt = T/K/kw;
    
    % compute in batch if memory overflows
    
    % Define the latent Gaussian Process
    f = C'*randn(n, nI) + mu;
    % Define the warped Gaussian Process
    z = warpfunc(hyp.dist,f);
    
    ZI = zeros(K,nI);
    for i=1:K
        ZI(i,:) = SimpsonRule(z((i-1)*kw+1:i*kw,:),dt);
    end
    % ZI = ZI + snI*randn(K,nI);
    ZI = K /T * ZI+snI*randn(K,nI);
   
end