clear all;close all;clc;

% f=@(x) x'*x;lb=[-1,-1]';ub=[1,1]';x0=[0.5,0.5]';
% [x,fval] = fmincon(f,x0,[],[],[],[],lb,ub) 
% [x,fval]= InteriorPoint(f,x0,lb,ub)

rng("default")
% Simulate Gaussian Process
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1/4; sf = 1; hyp.cov=log([ell; sf]);
q=0.6;
pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);

n = 200;
x = linspace(-10,10,n)';
f=SimGP(hyp,meanfunc,covfunc,x);

% Simulate Warped Gaussian Process
warpfunc=@(pd,p) invCdf(pd,p);
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);
% sum(z);

indexTest=1:5:n;
indexTrain=setdiff(1:n,indexTest);

Yhat=z(indexTrain);
Xtrain=x(indexTrain);
xstar=x(indexTest);
Rho=[0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99,1]';N=length(Rho);MSE=zeros(N,1);Accuracy=zeros(N,1);
for i=1:N
    rho=Rho(i);
    A=[rho,1-rho;1-rho,rho];
    Ypred=SBLUE(covfunc,hyp.cov,Yhat,Xtrain,xstar,A,q);
    Ytrue=z(indexTest);
    Ydiff=(Ypred-Ytrue)';
    MSE(i)=sum(Ydiff.^2)/length(Ydiff);
    Accuracy(i)=sum(Ydiff==0)/length(Ydiff);
    display("Iteration "+i+" rho="+rho+" MSE="+MSE(i)+" Accuracy="+Accuracy(i))
end
figure()
plot(Rho,MSE,'r',Rho,Accuracy,'b')
legend("MSE",'Accuracy')
% display("Data Generated")



% 
% 
% pd=makedist('Gamma','a',5,'b',10);
% % pd=makedist("Beta",'a',1,'b',1);
% G=@(x) norminv(cdf(pd,x));
% n=5;x0=ones(5,1)*0.1;lb=zeros(n,1);ub=[];
% rng('shuffle');
% A=randn(n);
% Kinv=A*A'
% Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
% Qneg=@(x) -Q(x);
% [vhat,Qnval]=InteriorPoint(Qneg,x0,lb,ub)
% Qval=-Qnval
% d2Q=hessianQ(pd,Kinv,G,vhat)
% 
% warpinv=@(pd,p) invCdfWarp(pd,p);
% [Qval,vhat,A]=LaplaceApproximation(pd,Kinv,warpinv,x0,lb,ub)
% 
% A\eye(n)
% 
% 
% % mu = 0;
% sigma = 1;
% pd = makedist('Normal','mu',mu,'sigma',sigma);
% f=@(x) pdf(pd,x);
% h=0.00001;
% pd=makedist('Gamma','a',2,'b',4);
% % pd=makedist("Beta",'a',1,'b',1);
%x0=(0.1:0.5:3)';

% x=[x0,x0+h];
% y=f(x);
% % df=diff(y,1,2)/h-dnorm(x0)
% % 
% % size(df)
% % size(x0)
% % gradientDist(pd,x0)-dnorm(x0)
% % df=gradientDist(pd,x0);
% % size(df)
% 
% % [y,d2f]=hessianDist(pd,x0);
% % y
% % d2f-hessianNorm(x0)
% 
% pd=makedist('Gamma','a',1,'b',3);
% % pd=makedist("Beta",'a',1,'b',1);
% x0=(0.1:0.2:1)';
% G=@(x) norminv(cdf(pd,x));
% % y=G(x)
% % dG=diff(y,1,2)/h
% % gradientG(pd1,G,x0)
% 
% % result=hessianG(pd,G,x0)
% % y=gradientG(pd,G,x0);
% % yh=gradientG(pd,G,x0+h);
% % d2G=diff([y,yh],1,2)/h
% % result-d2G
% 
% h=0.000001;
% n=size(x0,1);
% A=randn(n);
% Kinv=A*A';
% Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
% 
% 
% d2Q=hessianQ(pd,Kinv,G,x0);
% 
% d2Qnum=zeros(n,n);
% dA=zeros(n,n);
% dB=zeros(n,n);
% y=gradientQ(pd,Kinv,G,x0);
% for i=1:n
%     delta=zeros(n,1);
%     delta(i)=h;
%     yh=gradientQ(pd,Kinv,G,x0+delta);
%     d2Qnum(:,i)=(yh-y)./h;
% %     dA(:,i)=(Ah-A)./h;
% %     dB(:,i)=(Bh-B)./h;
% end
% diffD2=d2Qnum-d2Q
% diffD2A=dA-d2A
% diffD2B=dB-d2B


% dQ=gradientQ(pd,Kinv,G,x0)
% 
% dQnum=zeros(n,1);
% y=Q(x0)
% for i=1:n
%     delta=zeros(n,1);
%     delta(i)=h;
%     yh=Q(x0+delta);
%     dQnum(i)=(yh-y)/h;
% end
% dQnum-dQ

% f=@(x) x.^2;
% df=@(x) 2*x;
% d2f=zeros(n,n);
% y=df(x0);
% for i=1:n
%     delta=zeros(n,1);
%     delta(i)=h;
%     yh=df(x0+delta);
%     d2f(:,i)=(yh-y)./h; 
% end
% d2f

% 
% [d2Q,d2A,d2B,d2B1,d2B2,d2B3]=hessianQ(pd,Kinv,G,x0);
% 
% d2Qnum=zeros(n,n);
% dA=zeros(n,n);
% dB=zeros(n,n);
% [y,A,B]=gradientQ(pd,Kinv,G,x0);
% for i=1:n
%     delta=zeros(n,1);
%     delta(i)=h;
%     [yh,Ah,Bh]=gradientQ(pd,Kinv,G,x0+delta);
%     d2Qnum(:,i)=(yh-y)./h;
%     dA(:,i)=(Ah-A)./h;
%     dB(:,i)=(Bh-B)./h;
% end
% diffD2=d2Qnum-d2Q
% diffD2A=dA-d2A
% diffD2B=dB-d2B




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
