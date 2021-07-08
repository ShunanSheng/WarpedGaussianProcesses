function z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)
    % Simulate Warped Gaussian Process at points x 
    %
    % Input: meanfunc, covfunc, warpfunc, hyp, x 
    % Output: z
    f=SimGP(hyp,meanfunc,covfunc,x);
    z=warpfunc(hyp.dist,f);
end
