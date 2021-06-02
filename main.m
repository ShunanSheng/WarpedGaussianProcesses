clear all,close all


% Setup for Spatial Random field
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 2; sf = 1; hyp.cov=log([ell; sf]);
pd=makedist("Binomial",'N',1,'p',0.25); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);

% Setup for Temproal processes
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 = 1/2; sf0 = 1; hyp0.cov=log([ell0; sf0]);
meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1]);
% pd0=makedist('Normal','mu',3,'sigma',4);
pd0=makedist('Gamma','a',2,'b',4);
pd1=makedist('Gamma','a',2,'b',4);
T=10; 
hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T);



SP=struct("meanfunc",meanfunc,"covfunc",covfunc,"hyp",hyp);
H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

% [ZP,ZI,y]=SimData(SP,H0,H1);



M=20; K=50; snP=0.1; snI=0.1; 
t=linspace(0,hyp0.t,M)';

warpfunc=@(pd,p) invCdf(pd,p);

zI=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,0);

sumstats=@summaryMoment;
d=@distEuclid;
delta=1;
gamma=0.8;
yhat=NLRT(zI,H0,H1,warpfunc,K,snI,sumstats,d,delta,gamma)



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


























