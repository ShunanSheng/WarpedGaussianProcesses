function Cov_Y=covY(covfunc,hypcov,c,A,Xtrain)
    % Calculate Cov(Yhat,Yhat)
    %
    % Input : 
    % covfunc : the cov function of spatial field
    % hypcov : parameter of cov function 
    % A : transition matrix
    % c : invNormCdf(1-q)
    % Xtrain: the location of existing sensors
    %
    % Output:
    % Cov_Y : Cov(Yhat,Yhat)
    
    
    N=size(Xtrain,1);
    mY=meanY(A,c,N);
    p00=A(1);p10=A(2);p01=A(3);p11=A(4);
    
    Cov_Y=zeros(N); 
    T1=[-c -c];
    T2=[-c c];
    T3=[c -c];
    T4=[c c];
    for i=1:N - 1
        for j=i+1:N
            X_temp=[Xtrain(i) Xtrain(j)]';
            P1=mvncdf(T1,zeros(1,2),feval(covfunc{:},hypcov,X_temp));
            P2=mvncdf(T2,zeros(1,2),feval(covfunc{:},hypcov,X_temp));
            P3=mvncdf(T3,zeros(1,2),feval(covfunc{:},hypcov,X_temp));
            P4=mvncdf(T4,zeros(1,2),feval(covfunc{:},hypcov,X_temp));
            Cov_Y(i,j)=p11^2*P1+p10*p11*+p01*p11*P2+p01*p11*P3+p01^2*P4;
        end
    end
    Cov_Y = Cov_Y + Cov_Y';
    % Replace the diagonal elements
    % The varY is acturally E(Y_i^2)
    varY=(p01^2*normcdf(c)+p11^2*normcdf(-c)).*ones(N,1);
    Cov_Y(1:N + 1:end) = varY;
    Cov_Y=Cov_Y-mY*mY';
end