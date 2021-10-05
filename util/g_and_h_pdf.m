function p=g_and_h_pdf(x, g, h, loc, sca, tol)
%G_AND_H_CDF The cumulative distribution function of the g-and-h
%distribution
% Inputs:
%       x: input to the cdf
%       g: the g parameter
%       h: the h parameter
%       loc: the location parameter, default is 0
%       sca: the scale parameter, default is 1
%       tol: the numerical tolerance, default is 1e-8
% Outputs:
%       p: the density
    

if ~exist('loc', 'var') || isempty(loc)
    loc = 0;
end

if ~exist('sca', 'var') || isempty(sca)
    sca = 1;
end

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-8;
end


[z, ~] = g_and_h_inverse((x - loc) / sca, g, h, tol);
p=normpdf(z)./sca./grad_g_and_h(z,g, h, 0, 1);


end
