function m=meanY(p,c,N)
    % Calculate E(Yhat)
    %
    % Input : 
    % p : the modified confusion probability
    % c : invNormCdf(1-q)
    % N : the length
    %
    % Output:
    % m: E(Yhat)
    m=(p.p11.*normcdf(-c)+p.p01.*normcdf(c)).*ones(N,1);
end