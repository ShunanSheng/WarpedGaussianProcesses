clear all,close all,clc


% Setup for Spatial Random field
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 2; sf = 1; hyp.cov=log([ell; sf]);
pd=makedist("Binomial",'N',1,'p',0.25); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);

% Setup for Temproal processes
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 =1/2; sf0 = 1; hyp0.cov=log([ell0; sf0]);
meanfunc1 = @meanConst; 
covfunc1 = {@covSEiso}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% covfunc1 = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% pd0=makedist('Normal','mu',3,'sigma',4);
pd0=makedist('Normal','mu',0,'sigma',1);
% pd0=makedist('Gamma','a',1,'b',1);
pd1=makedist('Gamma','a',2,'b',4);
% pd1=makedist('Beta','a',1,'b',1);
T=10; M=20; K=20; snP=0.1; snI=0.1; 
lb0=zeros(M,1);ub0=[];lb1=zeros(M,1);ub1=[];
hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);



SP=struct("meanfunc",meanfunc,"covfunc",covfunc,"hyp",hyp);
H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

% [ZP,ZI,y]=SimData(SP,H0,H1);


options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);



% kw=100;K=20;yx=0;snI=0;n=kw*K;
% x=linspace(0,T,n)';
% C0 = chol(feval(covfunc0{:}, hyp0.cov, x)+1e-9*eye(n));
% mu0 = meanfunc0( hyp0.mean, x);
% 
% C1 = chol(feval(covfunc1{:}, hyp1.cov, x)++1e-9*eye(n));
% mu1 = meanfunc1( hyp1.mean, x);


% rng("default")
% zI1=SimFastIntData(hyp0,hyp1,C0,C1,mu0,mu1,warpfunc,kw,K,snI,yx);
% 
% zI2=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,yx);
% 
% diff=(zI1-zI2)'

% sumstats=@summaryMoment;
% d=@distEuclid;
% delta=1;
% gamma=0.8;
% yhat=NLRT(zI,H0,H1,warpfunc,K,snI,sumstats,d,delta,gamma)

% test on WGPLRT

% t=linspace(0,hyp0.t,M)';n=20;yn=rand(n,1)>0.5;gamma=1;yhat=2*ones(n,1);
% 
% 
% for i=1:n
%     zP=SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,yn(i));
%     yhat(i)=WGPLRT(zP,H0,H1,warpinv,t,snP,gamma);
%     display("Iteration "+i+",ground truth="+yn(i)+",prediction="+yhat(i))
% end
% diff=yn-yhat;
% figure()
% scatter(1:n,diff)
% sum(diff.^2)/n
% sum(diff==0)/n


% test on NLRT

sumstats=@summaryMoment;
d=@distEuclid;
delta=1;
gamma=1;

t=linspace(0,hyp0.t,M)';n=100;yn=rand(n,1)>0.5;yhat=zeros(n,1);
for i=1:n
    zI=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,yn(i));
    yhat(i)=NLRT(zI,H0,H1,warpfunc,K,snI,sumstats,d,delta,gamma);
    display("Iteration "+i+",ground truth="+yn(i)+",prediction="+yhat(i))
end
diff=yn-yhat;
figure()
scatter(1:n,diff)
Accuracy=sum(diff==0)/n



% Test some summary statistics
% zI0=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,0);
% zI1=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,1);
% figure();
% plot(t,zI0,'r',t,zI1,'b')
% legend("H0","H1")
% 
% mI0=mean(zI0)
% mI1=mean(zI1)
% mI20=moment(zI0,2)
% mI21=moment(zI1,2)
% 
% mode0=mode(zI0)
% mode1=mode(zI1)
% 
% k0=kurtosis(zI0)
% k1=kurtosis(zI1)
% 
% % figure()
% % subplot(1,2,1);
% % histogram(zI0,10)
% % subplot(1,2,2);
% % histogram(zI1,10)


























