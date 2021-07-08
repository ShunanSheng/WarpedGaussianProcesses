function d2f=hessianDist(pd,v)
    % Compute the second-order derivative of pdf of a given distribution;
    
    % Input:
    % pd: the probabilty distribution
    % v : the point to take gradient
    
    % Output: 
    % d2f : diagonal of Hessian of pdf of pd at v (n x 1) 
    
    h=0.00001;
    y=gradientDist(pd,v);
    yh=gradientDist(pd,v+h);
    d2f=diff([y,yh],1,2)/h;
end