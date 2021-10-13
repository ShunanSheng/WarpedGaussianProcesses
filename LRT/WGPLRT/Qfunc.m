function Qx=Qfunc(G,Kinv,pd,x)
    Qx=-1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
    if isnan(Qx)
        A= -1/2*G(x)'*Kinv*G(x);
        B= sum(log(gradientG(pd,G,x)));
        warning("NaN Objective")
    end
    dQ=gradientQ(pd,Kinv,G,x);
    if max(abs(dQ))>1000
        warning("gradient dQ too large")
    end

end
