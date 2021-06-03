function Lambda=testStats(A0,vhat0,Qval0,Klogdet0,A1,vhat1,Qval1,Klogdet1,snP,zP)
    n=size(zP);
    B0=A0+snP^2*eye(n);
    C0=A0\eye(n)+snP^2*eye(n);
    Cinv0=C0\eye(n);
    
    B1=A1+snP^2*eye(n);
    C1=A1\eye(n)+snP^2*eye(n);
    Cinv1=C1\eye(n);
    
    z0=zP-vhat0;
    z1=zP-vhat1;
    
    D=2*trace(log(B0))+Klogdet0-2*Qval0-2*trace(log(B1))-Klogdet1+2*Qval1;
    E=z0'*Cinv0*z0;
    F=z1'*Cinv1*z1;
    
    Lambda=1/2*(D+E-F);
end