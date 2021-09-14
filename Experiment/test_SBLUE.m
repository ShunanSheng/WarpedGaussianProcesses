%%% Test of SBLUE on 1D and 2D simulated data
clear all;close all;clc;

% Set up the spatial fied
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1; sf = 1; hyp.cov=log([ell; sf]);
q=0.8;
pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);
warpfunc=@(pd,p) invCdf(pd,p);


%% 1D data
N = 100;
x = linspace(-10,10,N)'; % Location of sensors
% Simulate Warped Gaussian Process
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);

%% 2D data
% Location of sensors 
n = 30; xinf=-10; xsup=10; N=n.^2;
[X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
xSp=reshape(X,[],1);
ySp=reshape(Y,[],1); 
x=[xSp,ySp];

% Generate the lantent binary spatial field
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,[xSp,ySp]);

%% Partition the training and test set
clc;
indexTest=1:5:N;
indexTrain=setdiff(1:N,indexTest);

Yhat=z(indexTrain);
Ytrue=z(indexTest);
Xtrain=x(indexTrain,:);
xstar=x(indexTest,:);

SBLUEprep=SBLUE_stats_prep(covfunc,hyp.cov,Xtrain,xstar,q); 
% the computation of P1,...,P4 is super slow

%%
clc;
% In the experiment,
% we may use Rho to control both true negative & true positive rate,
Rho=linspace(0,1,100)';
L=length(Rho);MSE=zeros(L,1);Accuracy=zeros(L,1);
TP=zeros(L,1);FP=zeros(L,1);

M=10000;
YT=repmat(Ytrue,[1,M]);
for i=1:L % We expect when rho is increasing, the performance will become better
    rho=Rho(i);
    A=[rho,1-rho;1-rho,rho]; % Define the transition matirx
    % Simulate the noisy data
    Yhat_noise=repmat( Yhat, [1,M] );
    for j=1:length(Yhat_noise(:))
        if rand()>rho
            Yhat_noise(j)=1-Yhat_noise(j);
        end
    end
    % Apply SBLUE
    SBLUE=SBLUE_stats(SBLUEprep,A,q);
    Ypred=SBLUE_pred(SBLUE,Yhat_noise);
    % Evaluate the MSE and Accuracy
    [tp,fp]=confusionMat(YT,Ypred);
    TP(i)=tp;
    FP(i)=fp;
    if mod(i,floor(L/10))==0
        disp("Iteration "+i+" rho="+rho+" TP="+tp+" FP="+fp)
    end
end


%%
close all;
figure();
plot(Rho,TP,"DisplayName","TP");
hold on;
plot(Rho,FP,"DisplayName","FP");
legend("TP","FP")
title("FP & TP plot vs Rho when q=",q);

