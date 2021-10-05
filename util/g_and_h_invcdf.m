function x=g_and_h_invcdf(p, g, h, loc, sca)
    % Find inverse cdf
    if ~exist('loc', 'var') || isempty(loc)
        loc = 0;
    end

    if ~exist('sca', 'var') || isempty(sca)
        sca = 1;
    end
    
    x=g_and_h(norminv(p), g, h, loc, sca);

end