
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
z=ZI(1,:);t=linspace(-10,10,1000)';pd_hat=makedist('Normal','mu',0,'sigma',sqrt(1/3*T^3/(K^3)));
y=pdf(pd_hat,t);
zf=z(z~=Inf);
var(zf)-1/3*T^3/(K^3)

close all;
figure();
histogram(z,'Normalization','pdf')
hold on;
plot(t,y);
title("Density plot vs histogram")
legend("Warpdist=N(0,1)")




%%
% C=feval(covfunc{:},hyp.cov,x)
figure()
plot(x,z)

%%
x=linspace(0,100,10000)';
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)';
y=pdf(pd,x);
close all;
figure();
histogram(z,'Normalization','pdf')
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
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)';
y=pdf(pd,x);
%%
close all;
figure();
histogram(z,'Normalization','probability')
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

