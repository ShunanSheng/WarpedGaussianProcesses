parpool(3)
parfor i=1:3, c(:,i) = eig(rand(1000)); end


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

