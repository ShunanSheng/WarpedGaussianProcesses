function f=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,yx)
    % Simulate latent temporal process at time points t 
    %
    % Input: 
    % hyp0,hyp1: parameters for the two hypotheses
    % meanfunc0,meanfunc1: mean functions for H0,H1
    % covfunc0,covfunc1  : covariance functions for H0,H1
    % warpfunc: the warp function handle
    % t : the time point
    % yx: binary value deciding H0 or H1
    %
    % Output: 
    % f: the temporal process

    if yx==0
        f=SimWGP(hyp0,meanfunc0,covfunc0,warpfunc,t);
    else
        f=SimWGP(hyp1,meanfunc1,covfunc1,warpfunc,t);
end