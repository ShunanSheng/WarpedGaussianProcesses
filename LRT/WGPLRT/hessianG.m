function d2G=hessianG(pd,G,v)
    % Compute the hessian of G at v
    % Output: n by 1 dig(hessian G) 
    h=@(x) pdf(pd,x);dh=@(x) gradientDist(pd,x);
    phi=@normpdf;dphi=@gradientNorm;
    w=G(v);
    
    numer=dh(v).*phi(w).^2-dphi(w).*h(v).^2;
    denom=phi(w).^3;
    d2G=numer./denom;
end