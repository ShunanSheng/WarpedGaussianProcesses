%%% Test of SBLUE on 1D and 2D simulated data
clear all;close all;clc;
% Set up the spatial fied
meanfunc = @meanConst;hyp.mean=0;
% covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp.cov=log([ell; sf]);
% covfunc = {@covFBM};sf0=1;h0=1/2;hyp.cov=[log(sf0);-log(1/h0-1)];
% covfunc = {@covMaterniso, 3}; ell=2; sf=1; hyp.cov=log([ell; sf]);
covfunc = {@covMaterniso, 3}; ell=exp(1); sf=1; hyp.cov=log([ell; sf]);
pd=[];c=0;
hyp=struct('mean',hyp.mean,'cov',hyp.cov,'dist',pd,'thres',c);
% warpfunc=@(pd,p) invCdf(pd,p);
warpfunc=@(c,x) indicator(c,x);

%% 1D data
% N = 1000;
% x = linspace(-100,100,N)'; % Location of sensors
% % Simulate Warped Gaussian Process
% g=SimGP(hyp,meanfunc,covfunc,x);
% z=warpfunc(c,g);
% z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);

%% 2D data
% % Location of sensors 
n = 30; xinf=-10; xsup=10; N=n.^2;
[X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
xSp=reshape(X,[],1);
ySp=reshape(Y,[],1); 
x=[xSp,ySp];

% Generate the lantent binary spatial field
g=SimGP(hyp,meanfunc,covfunc,x);
z=warpfunc(c,g);
% z=SimWGP(hyp,meanfunc,covfunc,warpfunc,[xSp,ySp]);

%% Partition the training and test set
indexTest=1:5:N;
indexTrain=setdiff(1:N,indexTest);

Ytrain=z(indexTrain);
Ytest=z(indexTest);
Xtrain=x(indexTrain,:);
Xtest=x(indexTest,:);

%% SBLUE prep
SBLUEprep=SBLUE_stats_prep(covfunc,meanfunc,hyp,Xtrain,Xtest); 

%% noisy SBLUE, assume p00=p11 in this case
rho=[0.9,0.9];lambda=[0.9,0.9];
A1=[rho(1),1-rho(1);1-lambda(1),lambda(1)];
A2=[rho(2),1-rho(2);1-lambda(2),lambda(2)];
xP=indexTrain(1:2:end);
xI=setdiff(indexTrain,xP);
liP=ismember(indexTrain,xP)';
liI=ismember(indexTrain,xI)';

M=1000;
YT=repmat(Ytest,[1,M]);
Yhat_noise=repmat( Ytrain, [1,M] );
% Generate the noisy 
for j=1:M
    rnd=rand(length(Ytrain),1);
    rnd1=rnd(liP)>rho(1);
    rnd2=rnd(liI)>rho(2);
    
    Yhat_noise(liP,j)=(1-rnd1).*Yhat_noise(liP,j)...
                                +rnd1.*(1-Yhat_noise(liP,j));
                            
    Yhat_noise(liI,j)=(1-rnd2).*Yhat_noise(liI,j)...
                                +rnd2.*(1-Yhat_noise(liI,j));                        
end

transitionMat=SBLUE_confusion(A1,A2,liP,liI);
% compute the adjusted confusion probability

% Apply SBLUE
SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c);
Ypred=SBLUE_pred(SBLUE,Yhat_noise);
% Evaluate the MSE and Accuracy
MSE_SBLUE_noisy=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));

