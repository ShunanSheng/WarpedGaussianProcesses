function dG=gradientG(pd,G,v)
    % Compute the gradient of G at v
    % Output: n by 1 dig(gradient G) 
    h=@(x) pdf(pd,x);phi=@normpdf;
    w=G(v);
    dG=h(v)./phi(w);
end