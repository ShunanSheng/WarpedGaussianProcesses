function y=binmcdf(XU,mu,Sigma)
    % A simplied version of Matlab function mvncdf without using the
    % statset and other fexible inputs
    % Evaluate cdf of a bivariate normal with mean mu and Variance Sigma
    % such that P(x1<=X1,x2<=X2)
 
    tol=1e-8;
    
    % Center the vectors
    s = sqrt(diag(Sigma));
    Rho = Sigma ./ (s*s');
    XU0 = XU - mu;
    XU0 = XU0./(s');
    
    
    rho=Rho(2);
    y = internal.stats.bvncdf(XU0, rho, tol);

end