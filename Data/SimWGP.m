function z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)
    % Simulate Warped Gaussian Process at points x 
    %
    % Input: meanfunc, covfunc, warpfunc, hyp, x 
    % Output: z
    f=SimGP(hyp,meanfunc,covfunc,x);
    
    str=func2str(warpfunc);
    str1='@(pd,p)invCdf(pd,p)';
    str2='@(c,x)indicator(c,x)';
    
    switch str
        case str1
            assert(isempty(hyp.dist)==false,"empty distribution");
            z=warpfunc(hyp.dist,f);
        case str2
            assert(isempty(hyp.thres)==false,"empty threshold");
            z=warpfunc(hyp.thres,f);
        otherwise
            error("undefined warping function");    
    end
        
end
