% Gaussian Process regression wo GPML library

% Define the mean and covariance function

mean=@meanFunc;
cov=@covFunc;

% Generate the synthetic dataset
% Y=sin3X+sigma^2I
sn=0.3;
n= 100;
% x = gpml_randn(0.15,n, 1);                 % 20 training inputs
x=linspace(-8,8,n)';
y = sin(3*x) + sn^2*gpml_randn(0.2,n, 1);  % 20 noisy training targets
xs = linspace(-3, 3, 61)';        % 61 test inputs 

% Select a subset of points, 
index=1:5:100;
X=x(index);Y=y(index);
xt=x(setdiff(1:length(x), index));yt=y(setdiff(1:length(y), index));
plot(x,y,'+')
hold on
plot(X,Y,'x')

N=length(X);
Nt=length(xt);
% GP regression
kXX=feval(cov,X,X);
kxX=feval(cov,xt,X);
kxx=feval(cov,xt,xt);

alpha=((kXX+sn.^2*eye(N))\kxX')';
mpost=alpha* Y;
vpost=kxx-alpha*kxX';
spost=sqrt(diag(vpost));
post_samples=chol(vpost+1e-9*eye(Nt))*gpml_randn(0.1,Nt,1)+mpost;


% plot(X,Y,'x')
% hold on
grid on
f = [mpost+2*spost;flip(mpost-2*spost,1)];
fill([xt; flipdim(xt,1)], f, [7 7 7]/8)
hold on 
plot(xt,mpost,'r')
hold on
plot(xt,yt,'+',xt,post_samples,'b')
title('Simple GP regression')





% x=randn(n,1);
% M=feval(mean,n);
% K=feval(cov,x,x);
% y=chol(K)'*randn(n,1)+M+sn^2*randn(n,1);


%%%%%%%%%%%%%%%
% Functional utilities
function k=SE(a,b)
    sigma=1;ell=0.45;
    k=sigma.^2*exp(-(a-b).^2/2/ell.^2);
end

function K=covFunc(a,b)
    K=zeros(length(a),length(b));
    for j=1:length(b)
        for i=1:length(a)
            K(i,j)=SE(a(i),b(j));
        end
    end
end

function m=meanFunc(n)
    m=zeros(n,1)
end

