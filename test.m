clear all;close all;clc;

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);
f=@(x) pdf(pd,x);
h=0.00001;
x0=(1:0.5:3)';
x=[x0,x0+h];
y=f(x);
% df=diff(y,1,2)/h-dnorm(x0)
% 
% size(df)
% size(x0)
% gradientDist(pd,x0)-dnorm(x0)
% df=gradientDist(pd,x0);
% size(df)

% [y,d2f]=hessianDist(pd,x0);
% y
% d2f-hessianNorm(x0)

pd=makedist('Gamma','a',2,'b',4);
% pd=makedist("Beta",'a',1,'b',1)
G=@(x) norminv(cdf(pd,x));
% y=G(x)
% dG=diff(y,1,2)/h
% gradientG(pd1,G,x0)
% 
% result=hessianG(pd,G,x0)
% y=gradientG(pd,G,x0);
% yh=gradientG(pd,G,x0+h);
% d2G=diff([y,yh],1,2)/h
% result-d2G

n=size(x0,1);
A=randn(n);
Kinv=A*A';
Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
dQ=gradientQ(pd,Kinv,G,x0)

dQnum=zeros(n,1);
y=Q(x0)
for i=1:n
    delta=zeros(n,1);
    delta(i)=h;
    yh=Q(x0+delta);
    dQnum(i)=(yh-y)/h;
end
dQnum-dQ

d2Q=hessianQ(pd,Kinv,G,x0)

% 
% f=@squareAll
% integral(f,0,1)
% 
% meanfunc = @meanConst; 
% covfunc = @covSEiso; ell = 1/4; sf = 1;
% sn=0.01;
% hyp=struct('mean',0,'cov',log([ell; sf]),'noise',log(sn))
% 
% n = 1000;
% x = linspace(-10,10,n)';
% f1=zeros(n,1);
% for i=1:n
%    f1(i)=SimGP(hyp,meanfunc,covfunc,x(i));
% end
% 
% f2=SimGP(hyp,meanfunc,covfunc,x);
% figure()
% plot(x,f1,'r',x,f2,'b');
% K=10;
% zI=zeros(K,1);
% for i=1:K
%     zI(i)=20/n*sum(f2((i-1)*n/K+1:i*n/K));
% end
% zI
% 
% 
% 
% function f=squareAll(x)
%     f=x.^2;
% end
% 
% 
% 
% 
% 
% 
% 
% % 
% % % Simulate Gaussian Process
% % meanfunc = @meanConst; 
% % covfunc = @covSEiso; ell = 1/2; sf = 1;
% % sn=0.01;
% % hyp=struct('mean',0,'cov',log([ell; sf]),'noise',log(sn))
% % 
% % n = 100;
% % x = linspace(-10,10,n)';
% % f=SimGP(hyp,meanfunc,covfunc,x);
% % 
% % % Simulate Warped Gaussian Process
% % pd=makedist("Binomial",'N',1,'p',0.25); % Bernouli(p)
% % warpfunc=@(p) invCdf(pd,p);
% % z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);
% 
% 
% 
% 
% % % Point Observations
% % zP=SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,M,snP,y(1))
% % % Integral Observations
% % zI=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,y(1))
% 
% 
% % f=@(t) SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,y(1));
% % result=f(1)
% 
% % Assume all point sensors collect values at the same time instants
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % g=reshape(g,[],10);
% % size(g)
% % 
% % 
% % figure();
% % surf(X,Y,g)
% 
% 
% % 
% % t=linspace(0,T,M)';
% % % f=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,y(1))
% % 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % figure();
% % plot(x,z)
% 
% 
% 
% 
% 
% 
% % pd=makedist("Binomial",'N',1,'p',0.25)% Bernouli(p)
% % Finv=@(x) icdf(pd,x);
% % p=linspace(0,1,100);
% % figure();
% % plot(p,Finv(p))
% 
% % pd=makedist('Normal');
% 
% 
% % 
% % pd=makedist('Normal');
% % Finv=@(x) icdf(pd,x);
% % p=[0,0.1,0.5,0.8,1]
% % Finv(p)
% % 
% % invCdf(pd,Finv(p))
% 
% 
% % 
% % A=covfunc(hyp.cov,x);
% % B=covFunc(x',x',ell);
% % chol(A);
% % chol(B);
% % C=A-B
% % 
% % function K=covFunc(X,Y,ell)
% %     D=pdist2(X',Y');
% %     K=exp(-D.^2/2/ell.^2);
% % end
