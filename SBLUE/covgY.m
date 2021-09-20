function Cov_g=covgY(p,Cov_xstar,c)
    % Calculate Cov(gsatr,Yhat)
    %
    % Input : 
    % p : the modified confusion probability
    % Cov_xstar : C(xstar,xtrain)
    % c : invNormCdf(1-q)
    %
    % Output:
    % Cov_g : Cov(gsatr,Yhat)
    
    nstar=size(Cov_xstar,1);
    P=repmat((p.p11-p.p01)',[nstar,1]);
    Cov_g=P.*Cov_xstar.*exp(-c^2/2)./sqrt(2*pi);
end