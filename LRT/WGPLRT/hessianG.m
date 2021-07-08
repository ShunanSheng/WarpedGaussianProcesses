function d2G=hessianG(pd,G,v)
    % Compute the hessian of G at v
    
    % Input:
    % pd: the probabilty distribution
    % G : the inverse warping function handle 
    % v : the point to take gradient
    
    % Output: 
    % d2G : diagonal of Hessian of G at v (n x 1)
    
    h=@(x) pdf(pd,x);dh=@(x) gradientDist(pd,x);
    phi=@normpdf;dphi=@gradientNorm;
    w=G(v);
    
    numer=dh(v).*phi(w).^2-dphi(w).*h(v).^2;
    denom=phi(w).^3;
    d2G=numer./denom;
end