function Qx=Qfunc(G,Kinv,pd,x)
    % Evalaute the value of function Q according to equation (10) in the
    % paper
    % Input:
    % pd: the probabilty distribution
    % Kinv : the inverse covariance matrix C(T_{1:M},T_{1:M})
    % G : the inverse warping function handle 
    % x : the point to evaluate
    
    % Output: 
    % Q : gradient of Q (n x 1)
    
    Qx=-1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
    
    if isnan(Qx)
%         A = -1/2*G(x)'*Kinv*G(x);
%         B = sum(log(gradientG(pd,G,x)));
        warning("NaN Objective")
    end
    dQ = gradientQ(pd,Kinv,G,x);
    if max(abs(dQ))>1000
        warning("gradient dQ too large")
    end

end
