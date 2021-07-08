function df=gradientDist(pd,v)
    % Calculate the gradient of cdf of the given distribution 
    % Input:
    % pd: the probabilty distribution
    % v : the point to take gradient
    % Output: 
    % df: the gradient of pd at v (n x 1)
    h=0.00001;
    V=[v,v+h];
    y=pdf(pd,V);
    df=diff(y,1,2)/h;
end