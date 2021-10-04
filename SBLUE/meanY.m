function m=meanY(p,chat)
    % Calculate E(Yhat)
    %
    % Input : 
    % p : the transition probability
    % chat : the normalzied threshold
    %
    % Output:
    % m: E(Yhat)


    m=p.p11.*normcdf(-chat)+p.p01.*normcdf(chat);
end