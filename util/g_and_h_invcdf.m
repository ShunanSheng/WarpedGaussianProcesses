function x=g_and_h_invcdf(p, g, h, loc, sca)
    % Find inverse cdf
    % Inputs:
    %       p: the probability 
    %       g: the g parameter
    %       h: the h parameter
    %       loc: the location parameter, default is 0
    %       sca: the scale parameter, default is 1
    % Outputs:
    %       x: the value of g_and_h at such that P(X<x)=p
    if ~exist('loc', 'var') || isempty(loc)
        loc = 0;
    end

    if ~exist('sca', 'var') || isempty(sca)
        sca = 1;
    end
    
    x=g_and_h(norminv(p), g, h, loc, sca);

end