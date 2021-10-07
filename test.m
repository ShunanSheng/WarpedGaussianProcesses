f=@(x) x.*exp(-x.^2./2);
c=1;
integral(f,c,Inf)
exp(-c^2/2)




%%
clc;clear
pd1=makedist("Laplace","mu",0,"sigma",1)
x=[-1,0,1,2]';
% pd1=makedist("Normal","mu",0,"sigma",1)
cdf(pd1,x)
df=gradientDist(pd1,x)

g=1;h=1;loc=10;sca=1;
[v, iter] = g_and_h_inverse(x, g, h)
[p, iter] = g_and_h_cdf(x, g, h, loc, sca)

pd=makedist("g_and_h","g",-0.5,"h",1,'loc',loc,'sca',sca)
cdf(pd,x)
cdf(pd,icdf(pd,[0.1,0.2]'))
pdf(pd,x)

df=gradientDist(pd,x)
% gradientDist(pd,x)


%%
pd=makedist("Normal")
pd=makedist("Laplace")

%%

p=g_and_h_pdf(x, g, h, loc, sca)

v=linspace(0,5,10)';
delta=0.00000001;
V=[v,v+delta];
% y=g_and_h(V, g, h, loc, sca)
y=exp(V.^2);
df=diff(y,1,2)/delta

diff=df-exp(v).*2.*v

dgh=grad_g_and_h(v, g, h, loc, sca)-df
% df=gradientDist(pd,v)


%% when h=g=0, the g_and_h transformation is effectively the identity
clc;clear

z=randn(10000,1);
g=0;h=1;loc=0;sca=1;
x=g_and_h(z, g, h, loc, sca);

g=0;h=0;loc=0;sca=1;
y=g_and_h(z, g, h, loc, sca);
disp(min(x))
disp(min(y))
figure()
histogram(z)
figure()
histogram(x)
hold on
histogram(y)

%%
diff=z-x;

[zx, iter] = g_and_h_inverse(x, g, h);
diff_z=zx-z;

fz=normcdf(z);
fx=g_and_h_cdf(z, g, h, loc, sca);
diff_f=fz-fx;
dgh=grad_g_and_h(z, g, h, loc, sca);

pz=normpdf(z);
px=g_and_h_pdf(x, g, h, loc, sca);
diff_p=px-pz;

invz=norminv(pz);
invx=g_and_h_invcdf(px, g, h, loc, sca);
diff_inv=invz-invx;


%%
% clc
C=zeros(length(Ytrain),1);
C(Ytrain==1)=1;
C(Ytrain==0)=0;
figure()
scatter3(Xtrain(:,1),Xtrain(:,2),Ytrain,[],C,'filled')


%%
clc,clear all;
load fisheriris
X = meas;
Y = species;

Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1)
Xnew = [min(X);mean(X);max(X)];
[label,score,cost] = predict(Mdl,Xnew)


%%
clc
f=@(x) x.*exp(-x.^2/2);
c=-10;
integral(f,c,Inf)-exp(-c^2/2)


%% Include ACF, PACF in the summary statistics in NLRT
clc;clear all;
rng('default') % For reproducibility
ncols=3;
e = randn(1000,ncols);
y = filter([1 -11],1,e);
lag=1;
acf=zeros(lag+1,ncols);
for col=1:ncols
    acf(:,col) = autocorr(y(:,col),lag);
end


%%
% Check validity of S-BLUE, mean and covariance
clear all;clc;
% Set up the spatial fied
meanfunc = @meanConst; hyp.mean=1;
% covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp.cov=log([ell; sf]);
% covfunc={@covFBM};sf0=1;h0=1/2;hyp.cov=[log(sf0);-log(1/h0-1)];
covfunc = {@covMaterniso, 3}; ell1=1/2; sf1=1; hyp.cov=log([ell1; sf1]);
q=0.5;
% pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
pd=[];c=0;
hyp=struct('mean',hyp.mean,'cov',hyp.cov,'dist',pd,'thres',c);
% warpfunc=@(pd,p) invCdf(pd,p);
warpfunc=@(c,x) indicator(c,x);


%% 1D data
N = 1000;
x = linspace(-10,10,N)'; % Location of sensors
% Simulate Warped Gaussian Process
v=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);

%% Partition the training and test set
clc;
indexTest=1:5:N;
indexTrain=setdiff(1:N,indexTest);

Yhat=v(indexTrain);
Ytrue=v(indexTest);
Xtrain=x(indexTrain,:);
xstar=x(indexTest,:);

SBLUEprep=SBLUE_stats_prep(covfunc,meanfunc,hyp,Xtrain,xstar) 
% the computation of P1,...,P4 is super slow

%% Use different A1 and A2 for SBLUE
clc;
rho=[1,1];lambda=[1,1];
A1 = [rho(1),1-rho(1);1-lambda(1),lambda(1)];
A2 = [rho(2),1-rho(2);1-lambda(2),lambda(2)];
xP=indexTrain(1:2:end);
xI=setdiff(indexTrain,xP);
liP=ismember(indexTrain,xP)';
liI=ismember(indexTrain,xI)';

M=1000;
YT=zeros(length(x),M);G_star=zeros(length(indexTest),M);
for j=1:M
    f=SimGP(hyp,meanfunc,covfunc,x);
%     YT(:,j)=warpfunc(hyp.dist,f);
    YT(:,j)=warpfunc(hyp.thres,f);
    G_star(:,j)=f(indexTest);
end


Yhat_noise=zeros(length(indexTrain),M);

for j=1:M
    rnd=rand(length(Yhat),1);
    rnd1=rnd(liP)>rho(1);
    rnd2=rnd(liI)>rho(2);
    
    Yhat_noise(liP,j)=(1-rnd1).*YT(liP,j)...
                                +rnd1.*(1-YT(liP,j));
                            
    Yhat_noise(liI,j)=(1-rnd2).*YT(liI,j)...
                                +rnd2.*(1-YT(liI,j));                        
end

transitionMat=SBLUE_confusion(A1,A2,liP,liI);
% compute the adjusted confusion probability

%%
% Apply SBLUE
SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c);
mY_empi=mean(Yhat_noise,2);
diff_mY=SBLUE.mY-mY_empi;
covY_empi=cov(Yhat_noise');
diff_covY=SBLUE.CovY-covY_empi;

covgY_empi_ori=cov([Yhat_noise;G_star]');
covgY_empi=covgY_empi_ori(length(indexTrain)+1:end,1:length(indexTrain));
diff_covgY=SBLUE.Covg-covgY_empi;

disp("Done")

% Ypred=SBLUE_pred(SBLUE,Yhat_noise);
% % Evaluate the MSE and Accuracy
% MSE=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:))




%%
clear all
plot(0:10,0:10)
text(5,5,'TEST TEXT')
%%
clc
rng('default')
Y=[1,0,1,0,1,1,1]';

index=logical([0,1,1,0,1,0,1]');
M=length(Y(index))
rnd=rand(M,1);
Yindex=Y(index)

Yindex1=Yindex;

for i=1:M
    x=rnd(i);
    if x>0.5
        Yindex1(i)=1-Yindex(i);
    end
end
Yindex1



x=rnd>0.5;

Y2(index)=(1-x).*Y(index)+x.*(1-Y(index));

Y2(index)




%%
mu = [0 0];
Sigma = 1*eye(2);
XU=[0 0];
rng('default')  % For reproducibility
binmcdf(XU,mu,Sigma)







%%
clear all; clc;

A=[1,2;2,1];
try chol(A);
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
    disp(A)
end

eig(A)

%%
parpool(3)
parfor i=1:3, c(:,i) = eig(rand(1000)); end

%% Verify model does generate the correct integral observation
% Assume K(t,s)=min(t,s), m=0, then f becomes a Brownian motion
% Let pd~N(0,1), then W=Id, so z is also the Brownian motion. 
% Let noise to be 0, the integral of z over [0,T/K] is N(0,1/3*T^3/K^3)
clear all;clc;

pd=makedist('Normal','mu',0,'sigma',1);
meanfunc = @meanConst; 
covfunc={@covFBM};sf=1;h=1/2;hyp.cov=[log(sf);-log(1/h-1)];
warpfunc=@(pd,p) invCdf(pd,p);
T=400;K=100;
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd,'t',T);



%% Check integral observations
clc;
nI=10000;snI=0;
kw= ceil(exp(log(1000000*T/K/180)/4)); % calculate the number of point neeed per window under Simpson's rule with 0.01 error
kw= round(kw/2)*2+1;n=kw*K;x=linspace(0,T,n)';
C = chol(feval(covfunc{:}, hyp.cov, x)+1e-16*eye(n));
mu = meanfunc( hyp.mean, x);
ZI=SimIntData(hyp,C,mu,warpfunc,K,kw,snI,nI);

%%
v=ZI(1,:);t=linspace(-10,10,1000)';pd_hat=makedist('Normal','mu',0,'sigma',sqrt(1/3*T^3/(K^3)));
y=pdf(pd_hat,t);
zf=v(v~=Inf);
var(zf)-1/3*T^3/(K^3)

close all;
figure();
histogram(v,'Normalization','pdf')
hold on;
plot(t,y);
title("Density plot vs histogram")
legend("Warpdist=N(0,1)")




%%
% C=feval(covfunc{:},hyp.cov,x)
figure()
plot(x,v)

%%
x=linspace(0,100,10000)';
v=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)';
y=pdf(pd,x);
close all;
figure();
histogram(v,'Normalization','pdf')
hold on;
plot(x,y);
title("Density plot vs histogram")
legend("Warpdist=N(0,1)")








%%
% Check Interior point method
Q=@(x) x.^2;
x0=2;
lb=[];
ub=[];
InteriorPoint(Q,x0,lb,ub)
%% 
clear all;clc;

pd=makedist('Normal','mu',2,'sigma',4);
% pd=makedist('Gamma','a',2,'b',4);
pd=makedist('tLocationScale','mu',-1,'sigma',1,'nu',3)
warpinv=@(pd,p) invCdfWarp(pd,p);
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell0 =1/2; sf0 = 1; hyp.cov=log([ell0; sf0]);
warpfunc=@(pd,p) invCdf(pd,p);
%% Check WGP does generate the correct distribution
T=10;
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd,'t',T);
x=linspace(0,50,10000)';
v=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)';
y=pdf(pd,x);
%%
close all;
figure();
histogram(v,'Normalization','probability')
hold on;
plot(x,y);
title("Density plot vs histogram")
legend("Warpdist=Gamma(2,4)")

%%
n=10;
v=linspace(0,1,n)';


K0=feval(covfunc{:}, hyp.cov, v);
Kchol0=chol(K0+1e-9*eye(n)); 
Kichol0=Kchol0\eye(n); 
Klogdet0=2*trace(log(Kchol0));
Kinv=Kichol0*Kichol0';


G=@(x) warpinv(pd,x);
Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));


%%
% test for dG, d2G, dQ, d2Q
Gv=G(v)
dG=gradientG(pd,G,v)
d2G=hessianG(pd,G,v)
dGhat=gradientEmpirical(G,v)
diffdG=diag(dGhat)-dG

Qv=Q(v)
dQ=gradientQ(pd,Kinv,G,v)
dQhat=gradientEmpirical(Q,v)'
diffdQ=dQhat-dQ

d2Q=hessianQ(pd,Kinv,G,v)

F=@(x) gradientQ(pd,Kinv,G,x);
d2Qhat=gradientEmpirical(F,v)

diffd2Q=d2Qhat-d2Q


%% Verify the LA
% Let v be a 1x1 vector
% Kinv=eye(1);
x0=0.1;
lb=[];ub=[];
[Qval,vhat,A]=LaplaceApproximation(pd,Kinv,warpinv,x0,lb,ub)
%%
G=@(x) warpinv(pd,x);
Q=@(x) -1/2*G(x)'*Kinv*G(x)+sum(log(gradientG(pd,G,x)));
lv=linspace(-10,10,100)';
lQv=zeros(100,1);
for i=1:100
    lQv(i)=Q(lv(i));
end
figure()
plot(lv,lQv)

%%

function dF=gradientEmpirical(F,v)
    n=size(v,1);
    o=zeros(n,1);
    h=0.0001;
    dF=[];
    for i=1:n
       delta=o;
       delta(i)=h;
       vhat=v+delta;
       dF=[dF,(F(vhat)-F(v))/h];
    end
end

