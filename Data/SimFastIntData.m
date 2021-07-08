function zI=SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,kw,K,snI,yx)
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
    
    n=kw*K;
    % Define the latent Gaussian Process
    f = C'*randn(n, 1) + mu;
    % Define the warped Gaussian Process
    z=warpfunc(hyp.dist,f);
    
    zI=zeros(K,1);
    for i=1:K
        zI(i)=T/K/kw*sum(z((i-1)*kw+1:i*kw)); % Take riemann-stiejtes sum
    end
    zI=zI+snI*randn(K,1);
end