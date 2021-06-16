function df=gradientDist(pd,v)
%  Calculate the gradient of cdf of the given distribution 
    h=0.00001;
    V=[v,v+h];
    y=pdf(pd,V);
    df=diff(y,1,2)/h;
end