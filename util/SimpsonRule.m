function result=SimpsonRule(z,dt)
    % Use Simpson's rule to evaluate the integral integral z dt over
    % [(k-1)*T/K, k*T/K]
    
    % Input: 
    % z : function values in matrix form kw*K x nI
    % dt : step size
    % Output:
    % result
    
    result = dt./3.*(z(1,:)+2.*sum(z(3:2:end-2,:),1)+4.*sum(z(2:2:end,:),1)+z(end,:));
    
end