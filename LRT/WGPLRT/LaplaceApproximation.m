function [Qval,vhat,A]=LaplaceApproximation(pd,Kinv,warpinv,x0,lb,ub)
    G=@(x) warpinv(pd,x);
    Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
    Qneg=@(x) -Q(x);
    [vhat,Qnval]=InteriorPoint(Qneg,x0,lb,ub);
    Qval=-Qnval;
    A=-hessianQ(pd,Kinv,G,vhat); % negative hessian or hessian
end

