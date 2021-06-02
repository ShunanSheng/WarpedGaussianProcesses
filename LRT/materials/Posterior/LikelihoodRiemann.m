% Vaidate the approximation of marginal likelihood function
% Input: Z,A,sigma2,vhat,M=2
clc;
clear all;
M=2;
A=[1,0;0,1];
sigma=0.1;
vhat=[0.1,0]';
z=[1,0]';

% l=3*sigma;

n=1000;
l=5*sigma;
epsilon_step1=linspace(-l,l,n);
epsilon_step2=linspace(-l,l,n);
% [epsilon1,epsilon2]=meshgrid(epsilon_step1,epsilon_step2);
% C=meshgrid(epsilon_step1,epsilon_step2);

% Evaluate via Riemann Sum
Fval=@(x1,x2) F1(x1,x2,A,z,vhat,sigma);
Y=0;
for i=1:n
    epsilon1=epsilon_step1(i);
    for j=1:n
        epsilon2=epsilon_step2(j);
        Y=Y+Fval(epsilon1,epsilon2);
    end
end
integral1=Y*4*l.^2/n.^2

% From derivation
C=(2*pi)^(M/2)*det(A+sigma^(-2)*eye(M))^(-1/2);
integral2=C*exp(-(z-vhat)'*inv((inv(A)+sigma^2*eye(M)))*(z-vhat)/2)



function Fval=F1(x1,x2,A,z,vhat,sigma)
    x=[x1;x2];
    Fval=exp(-(z-x-vhat)'*A*(z-x-vhat)./2-sigma^(-2)*x'*x./2);
end






