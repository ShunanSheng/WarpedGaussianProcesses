function Cov_g=covgY(A,Cov_xstar,c)
    % Calculate Cov(gsatr,Yhat)
    %
    % Input : 
    % A : transition matrix
    % Cov_xstar : C(xstar,xtrain)
    % c : invNormCdf(1-q)
    %
    % Output:
    % Cov_g : Cov(gsatr,Yhat)
    
    p11=A(4);p01=A(3);
    Cov_g=(p11-p01).*Cov_xstar.*exp(-c^2/2)./sqrt(2*pi);
end