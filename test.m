
f=@squareAll
integral(f,0,1)

meanfunc = @meanConst; 
covfunc = @covSEiso; ell = 1/4; sf = 1;
sn=0.01;
hyp=struct('mean',0,'cov',log([ell; sf]),'noise',log(sn))

n = 1000;
x = linspace(-10,10,n)';
f1=zeros(n,1);
for i=1:n
   f1(i)=SimGP(hyp,meanfunc,covfunc,x(i));
end

f2=SimGP(hyp,meanfunc,covfunc,x);
figure()
plot(x,f1,'r',x,f2,'b');
K=10;
zI=zeros(K,1);
for i=1:K
    zI(i)=20/n*sum(f2((i-1)*n/K+1:i*n/K));
end
zI



function f=squareAll(x)
    f=x.^2;
end







% 
% % Simulate Gaussian Process
% meanfunc = @meanConst; 
% covfunc = @covSEiso; ell = 1/2; sf = 1;
% sn=0.01;
% hyp=struct('mean',0,'cov',log([ell; sf]),'noise',log(sn))
% 
% n = 100;
% x = linspace(-10,10,n)';
% f=SimGP(hyp,meanfunc,covfunc,x);
% 
% % Simulate Warped Gaussian Process
% pd=makedist("Binomial",'N',1,'p',0.25); % Bernouli(p)
% warpfunc=@(p) invCdf(pd,p);
% z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);




% % Point Observations
% zP=SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,M,snP,y(1))
% % Integral Observations
% zI=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,y(1))


% f=@(t) SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,y(1));
% result=f(1)

% Assume all point sensors collect values at the same time instants



















% g=reshape(g,[],10);
% size(g)
% 
% 
% figure();
% surf(X,Y,g)


% 
% t=linspace(0,T,M)';
% % f=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,y(1))
% 


























% figure();
% plot(x,z)






% pd=makedist("Binomial",'N',1,'p',0.25)% Bernouli(p)
% Finv=@(x) icdf(pd,x);
% p=linspace(0,1,100);
% figure();
% plot(p,Finv(p))

% pd=makedist('Normal');


% 
% pd=makedist('Normal');
% Finv=@(x) icdf(pd,x);
% p=[0,0.1,0.5,0.8,1]
% Finv(p)
% 
% invCdf(pd,Finv(p))


% 
% A=covfunc(hyp.cov,x);
% B=covFunc(x',x',ell);
% chol(A);
% chol(B);
% C=A-B
% 
% function K=covFunc(X,Y,ell)
%     D=pdist2(X',Y');
%     K=exp(-D.^2/2/ell.^2);
% end
