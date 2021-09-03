function Lambda=testStats(A0,vhat0,Qval0,Klogdet0,A1,vhat1,Qval1,Klogdet1,snP,zP)
    % Calcualte the test statistic as in Eqn. 14
    %
    % Input: 
    % zP   : the point observations
    % A0,vhat0,Qval0: output of Laplace Approximation given null hypothesis
    % A1,vhat1,Qval1: output of Laplace Approximation given altenative hypothesis
    % Klogdet0, Klogdet1 : logdet of K0, K1
    % snP  : noise for point observations
    %
    % Output: 
    % Lambda: the value of the test statistic
    
    
%     n=size(zP);
    n = size(A0, 1);
    B0=A0+snP^2*eye(n);
    C0=A0\eye(n)+snP^2*eye(n);
    Cinv0=C0\eye(n);
    
    B1=A1+snP^2*eye(n);
    C1=A1\eye(n)+snP^2*eye(n);
    Cinv1=C1\eye(n);
    
    z0=zP-vhat0;
    z1=zP-vhat1;
    
    D=log(det(B0))+Klogdet0-2*Qval0-log(det(B1))-Klogdet1+2*Qval1;
    E=z0'*Cinv0*z0;
    F=z1'*Cinv1*z1;
    
    
%     b0=log(det(B0))
%     b1=log(det(B1))

    Lambda=1/2*(real(D+E-F));
    
end