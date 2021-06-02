function d2Q=hessianQ(pd,Kinv,G,v)
    % Compute the hessian of Q at v
    % Output: n by n hessian Q
    h=@(x) pdf(pd,x);dh=@(x) gradientDist(pd,x);d2h=@(x) hessianDist(pd,x);
    phi=@normpdf;dphi=@gradientNorm;d2phi=@hessianNorm;
    w=G(v);dG=gradientG(pd,G,v);d2G=hessianG(pd,G,v);
    
    A=-diag(Kinv*w.*d2G)-Kinv.*(dG*dG');
    
    B1=(d2h(v).*h(v)-dh(v).^2)./(h(v).^2);
    B2=-(d2phi(w).*h(v).^2+dh(v).*dphi(w).*phi(w))./(phi(w).^3);
    B3=2.*(h(v).^2.*dphi(w).^2)./(phi(w).^4);
    B=diag(B1+B2+B3);
    
    d2Q=A+B;
end
