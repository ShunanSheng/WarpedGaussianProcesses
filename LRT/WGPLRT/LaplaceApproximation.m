function [Qval,vhat,A]=LaplaceApproximation(pd,Kinv,warpinv,x0,lb,ub)
    % Find Laplace approximation of Q
    
    % Input:
    % pd: the probabilty distribution
    % Kinv : the inverse covariance matrix C(T_{1:M},T_{1:M})
    % warpinv : the inverse warping function handle G=warpinv(pd,x)
    % x0 : initial point
    % lb : lower bound of range(W), a vector
    % ub : upper bound of range(W), a vector
    
    % Output: 
    % Qval : Q(vhat)
    % vhat : the mode of Q
    % A    : -hessian(Qvat)

    % Initialize the function
    G=@(x) warpinv(pd,x);
    Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
    Qneg=@(x) -Q(x);
    
    % Use Interior point to find the mode and maximum value of Q
    [vhat,Qnval]=InteriorPoint(Qneg,x0,lb,ub);
    Qval=-Qnval;
    A=-hessianQ(pd,Kinv,G,vhat); % negative hessian or hessian
    
end

