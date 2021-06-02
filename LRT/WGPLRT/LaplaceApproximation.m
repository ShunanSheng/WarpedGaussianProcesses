function LaplaceApproximation(pd,lb,ub,Kinv,x0)
    G=@(x) norminv(cdf(pd,x));
    n=5;x0=ones(n,1)*0.1;lb=zeros(n,1);ub=[];
    rng('shuffle');
    A=randn(n);
    Kinv=A*A'
    Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
    Qneg=@(x) -Q(x);
    [vhat,Qnval]=InteriorPoint(Qneg,x0,lb,ub)
    Qval=-Qnval
    d2Q=hessianQ(pd,Kinv,G,vhat)
end