function Cov_Y=covY(A,CovP,mY,c)
    % Calculate Cov(Yhat,Yhat)
    %
    % Input : 
    % K : the converiance matrix of 
    % A : transition matrix
    % c : invNormCdf(1-q)
    %
    % Output:
    % Cov_Y : Cov(Yhat,Yhat)
    
    
    N=size(mY,1);
    p01=A(3);p11=A(4);
    
    % Get the materails for CovY
    P1=CovP.P1;
    P2=CovP.P2;
    P3=CovP.P3;
    P4=CovP.P4;
    
    % Evaluate E(Y^2)
    Cov_Y=p11^2.*P1+p01*p11.*P2+p01*p11.*P3+p01^2.*P4;
    Cov_Y = Cov_Y + Cov_Y';
    
    % Replace the diagonal elements
    % The current varY :=E(Y_i^2)
    varY=(p01^2*normcdf(c)+p11^2*normcdf(-c)).*ones(N,1);
    Cov_Y(1:N + 1:end) = varY;
    
    % Subtract E(Y)^2
    Cov_Y=Cov_Y-mY*mY';
end