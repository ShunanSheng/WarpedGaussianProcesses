function df=gradientNorm(v)
    % Compute the gradient of pdf of a standard normal N(0,1)
    % Input:
    % v : the point to take gradient
    
    % Output: 
    % df : gradient of Normal pdf at v (n x 1)

    df=normpdf(v).*(-v);
end