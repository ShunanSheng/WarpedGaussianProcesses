function dG=gradientG(pd,G,v)
    % Compute the gradient of function G=inverse of invCDF warping W at v
    % G(x)= invNormCdf(cdfPd(x))
    % Input:
    % pd: the probabilty distribution
    % G : the inverse warping function handle 
    % v : the point to take gradient
    
    % Output: 
    % dG : gradient of G at v (n x 1)
    h=@(x) pdf(pd,x);phi=@normpdf;
    w=G(v);
    dG=h(v)./phi(w);
end