function d2f=hessianNorm(v)
    % Hessian of pdf of Norm(0,1)
    % Input:
    % v : the point to take gradient
    
    % Output: 
    % d2f : diagonal of Hessian (nx1)
    d2f=normpdf(v).*(v.^2-1);
end