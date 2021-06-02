function dQ=gradientQ(pd,Kinv,G,v)
    % Compute the gradient of Q at v
    % Output: n by 1 gradient Q
    h=@(x) pdf(pd,x);dh=@(x) gradientDist(pd,x);
    phi=@normpdf;dphi=@gradientNorm;
    w=G(v);dG=gradientG(pd,G,v);
    
    A=-Kinv*w.*dG;
    numer=dh(v).*phi(w).^2-dphi(w).*h(v).^2;
    denom=phi(w).^2.*h(v);
    B=numer./denom;
    
    dQ=A+B;
end