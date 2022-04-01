function Cov_g=covgY(p,Cov_xstar,chat,Sigma)
    % Calculate Cov(gsatr,Yhat)
    %
    % Input : 
    % p : the modified confusion probability
    % Cov_xstar : C(xstar,xtrain)
    % chat : the normalzied threshold
    % Sigma : the std of xtrain, i.e, dif(Cov_xtrain)
    %
    % Output:
    % Cov_g : Cov(gsatr,Yhat)
    
    nstar=size(Cov_xstar,1);
    P=repmat((p.p11-p.p01)',[nstar,1]);
    C=repmat((chat.^2)',[nstar,1]);
    S=repmat(Sigma',[nstar,1]);
    Cov_g=P.*Cov_xstar.*exp(-C./2)./sqrt(2*pi)./S;
end