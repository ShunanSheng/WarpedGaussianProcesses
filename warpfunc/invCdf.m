function z=invCdf(pd,x)
    % inverse CDF warping, e.g. W
    z=icdf(pd,normcdf(x));
end