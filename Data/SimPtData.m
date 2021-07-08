function zP=SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,yx)
    % Simulate point observations at time points t 
    %
    % Input: 
    % hyp0,hyp1: parameters for the two hypotheses
    % meanfunc0,meanfunc1: mean functions for H0,H1
    % covfunc0,covfunc1  : covariance functions for H0,H1
    % warpfunc: the warp function handle
    % t  : the time points for which the sensor takes observation
    % snP: noises for point observations
    % yx : binary value deciding H0 or H1
    %
    % Output: 
    % zP: the integral observations
    
    M=size(t,1);
    zP=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,yx);
    zP=zP+snP*randn(M,1);
end
