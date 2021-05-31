function SimData(H0,H1,hyp,hyp0,hyp1)
    
% Binary spatial field GP(0,C), with SE kernel
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 2; sf = 1; hyp.cov=log([ell; sf]);

n = 5; xinf=-10; xsup=10;
[X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
xSp=reshape(X,[],1);
ySp=reshape(Y,[],1); 
pd=makedist("Binomial",'N',1,'p',0.25); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);

warpfunc=@(pd,p) invCdf(pd,p);
y=SimWGP(hyp,meanfunc,covfunc,warpfunc,[xSp,ySp]);

N=length(y); % Total number of sensors
% The indexes of point sensors and integral sensors
xI=1:5:n^2;NI=length(xI);
xP=setdiff(1:n^2,xI);NP=length(xP);

% Hypotheses for Temproal Process
T=10; 
meanfunc0 = @meanConst; 
covfunc0 = {@covSEiso}; ell0 = 2; sf0 = 1; hyp0.cov=log([ell0; sf0])
meanfunc1 = @meanConst; 
covfunc1 = {@covSEiso}; ell1=1/2; sf1=1; hyp1.cov=log([ell1; sf1])

pd0=makedist('Gamma','a',2,'b',4);
pd1=makedist('Gamma','a',1,'b',10);
hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T);

% without loss of generality we may assume M=K in practice
M=20; K=20; snP=0.1; snI=0.1;

% The Point Observations
ZP=zeros(M,NP);t=linspace(0,hyp0.t,M)';
for i=1:NP
   ZP(:,i)= SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,y(xP(i))); 
end
% The Integral Observations
ZI=zeros(M,NI);
for i=1:NI
   ZI(:,i)= SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,y(xI(i))); 
end

end