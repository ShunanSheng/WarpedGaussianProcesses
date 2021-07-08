function dQ=gradientQ(pd,Kinv,G,v)
    % Compute the gradient of Q at v
    % Q=-1/2 * G(v)'*Kinv*G(v)+sum(log(dG))
    % See detailed derivation in the implementation file
    % Input:
    % pd: the probabilty distribution
    % Kinv : the inverse covariance matrix C(T_{1:M},T_{1:M})
    % G : the inverse warping function handle 
    % v : the point to take gradient
    
    % Output: 
    % dQ : gradient of Q at v (n x 1)
    
    h=@(x) pdf(pd,x);dh=@(x) gradientDist(pd,x);
    phi=@normpdf;dphi=@gradientNorm;
    w=G(v);dG=gradientG(pd,G,v);
    
    A=-Kinv*w.*dG;
    numer=dh(v).*phi(w).^2-dphi(w).*h(v).^2;
    denom=phi(w).^2.*h(v);
    B=numer./denom;
    
    dQ=A+B;
end