function df=gradientNorm(v)
% Compute the gradient of pdf of a standard normal N(0,1)
    df=normpdf(v).*(-v);
end