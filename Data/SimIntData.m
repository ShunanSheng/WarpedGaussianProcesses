function zI=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,yx)
    % Simulate integral observations over time period [0,T] with K windows
    % using the given cov, mean function handles
    %
    % Input: 
    % hyp0,hyp1: parameters for the two hypotheses
    % meanfunc0,meanfunc1: mean functions for H0,H1
    % covfunc0,covfunc1  : covariance functions for H0,H1
    % warpfunc: the warp function handle
    % K  : total numebr of windows over [0,T]
    % snI: noises for integral observations
    % yx : binary value deciding H0 or H1
    %
    % Output: 
    % zI : the integral observations

    


    zI=zeros(K,1);T=hyp0.t;n0=100; 
    t=linspace(0,T,K*n0)'; % find a dense grid on [0,T]
    z=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,yx); % Simulate the latent temporal process
    for i=1:K
        zI(i)=T/K/n0*sum(z((i-1)*n0+1:i*n0)); % Take riemann-stiejtes sum
    end
    zI=zI+snI*randn(K,1);
end