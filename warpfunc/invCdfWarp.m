function z=invCdfWarp(pd,x)
    % inverse of invCdf warping, e.g. G
    z=norminv(cdf(pd,x));
end