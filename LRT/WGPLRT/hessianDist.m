function d2f=hessianDist(pd,v)
    % Compute the second-order derivative of pdf of a given distribution;
    % Return a n by 1 vector
    h=0.00001;
    y=gradientDist(pd,v);
    yh=gradientDist(pd,v+h);
    d2f=diff([y,yh],1,2)/h;
end