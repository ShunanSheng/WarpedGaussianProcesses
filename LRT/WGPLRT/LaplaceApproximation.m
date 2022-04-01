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
    Q=@(x)  Qfunc(G,Kinv,pd,x);
    Qneg=@(x) -Q(x);
    
    if isempty(lb) && isempty(ub)
        options=optimoptions('fminunc','Display','iter');
        [vhat,Qnval] = fminunc(Qneg,x0,options);
    else
        % Use Interior point to find the mode and maximum value of Q
        options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',50000);
        [vhat,Qnval]=InteriorPoint(Qneg,x0,lb,ub,options);
    end 
    Qval=-Qnval;
    A=-hessianQ(pd,Kinv,G,vhat); % negative hessian or hessian
    if max(abs(gradientQ(pd,Kinv,G,vhat)),[],'all')>0.1
        warning("vhat is not local maxmimum")
    end
end

