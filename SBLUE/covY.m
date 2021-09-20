function Cov_Y=covY(p,CovP,mY,c)
    % Calculate Cov(Yhat,Yhat)
    %
    % Input : 
    % p : the modified confusion probability
    % CovP: structure stores P1,P2,P3,P4
    % mY : mean vector of Yhat
    % c : invNormCdf(1-q)
    %
    % Output:
    % Cov_Y : Cov(Yhat,Yhat)
    
    
    N=size(mY,1);
    
    % Get the raw information for CovY, i.e. P(gi<c,gj<c),P(gi<c,gj>c),etc
    P1=CovP.P1;
    P2=CovP.P2;
    P3=CovP.P3;
    P4=CovP.P4;
    
    % Evaluate E(Y^2)
    p01=p.p01;p11=p.p11;
    p1=p01*p01';
    p2=p01*p11';
    p3=p11*p01';
    p4=p11*p11'; 
    
    
    Cov_Y=p1.*P1+p2.*P2+p3.*P3+p4.*P4;
    Cov_Y = Cov_Y + Cov_Y';
    
    % Replace the diagonal elements
    % The current varY :=E(Y_i^2)
    
    varY=(diag(p1).*normcdf(c)+diag(p4).*normcdf(-c)).*ones(N,1);
    Cov_Y(1:N + 1:end) = varY;
    
    % Subtract E(Y)^2
    Cov_Y=Cov_Y-mY*mY';

    
end

