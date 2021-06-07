function m=meanY(A,c,N)
    % Calculate E(Y)
    p11=A(4);p01=A(3);
    m=(p11*normcdf(-c)+p01*normcdf(c)).*ones(N,1);
end