function Cov_g=covgY(A,Cov_xstar,c)
    % Cov_xstar:=cov(x*,x_train)
    p11=A(4);p01=A(3);
    Cov_g=(p11-p01).*Cov_xstar.*exp(-c^2/2)./sqrt(2*pi);
end