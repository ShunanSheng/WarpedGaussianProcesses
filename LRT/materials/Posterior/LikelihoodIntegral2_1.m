clear all;
clc;
rng('shuffle');

M=2;
A = randn(2);
A=A' * A;
sigma=rand;
vhat=randn(2, 1);
z=randn(2, 1);


l=sigma * 6;

% Evaluate via Riemann Sum
func = @(x1, x2)(test_integrand({x1, x2}, A, z, vhat, sigma));

integral_numerical = integral2(func, -l, l, -l, l);

% From derivation
C=(2*pi)^(M/2)*det(A+sigma^(-2)*eye(M))^(-1/2);
integral_theoretical = C*exp(-(z-vhat)'*inv((inv(A)+sigma^2*eye(M)))*(z-vhat)/2);

fprintf('theoretical: %.10f, numerical: %.10f, difference: %.10f\n', ...
    integral_theoretical, integral_numerical, ...
    abs(integral_theoretical - integral_numerical));


