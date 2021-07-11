%%% Test for SBLUE on 1D and 2D simulated data

clear all;close all;clc;

% Set up the spatial fied
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp.cov=log([ell; sf]);
q=0.8;
pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);
warpfunc=@(pd,p) invCdf(pd,p);


%% 1D data
N = 400;
x = linspace(-10,10,N)';
% Simulate Warped Gaussian Process
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);


%% 2D data

% Location of sensors 
n = 20; xinf=-10; xsup=10; N=n.^2;
[X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
xSp=reshape(X,[],1);
ySp=reshape(Y,[],1); 
x=[xSp,ySp];

% Generate the lantent binary spatial field
z=SimWGP(hyp,meanfunc,covfunc,warpfunc,[xSp,ySp]);

%% Partition the training and test set

indexTest=1:5:N;
indexTrain=setdiff(1:N,indexTest);

Yhat=z(indexTrain);
Ytrue=z(indexTest);
Xtrain=x(indexTrain,:);
xstar=x(indexTest,:);


% Rho is the true negative & true positive rate
% Rho=[0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]';
Rho=[1];
L=length(Rho);MSE=zeros(L,1);Accuracy=zeros(L,1);

for i=1:L
    rho=Rho(i);
    A=[rho,1-rho;1-rho,rho]; % Define the transition matirx
    % Simulate the noisy data
    Yhat2=Yhat;
    for j=1:length(Yhat)
        if rand()>rho
            Yhat2(j)=1-Yhat(j);
        end
    end
    % Apply SBLUE
    Ypred=SBLUE(covfunc,hyp.cov,Yhat2,Xtrain,xstar,A,q);
    
    % Evaluate the MSE and Accuracy
    Ydiff=(Ypred-Ytrue)';
    MSE(i)=sum(Ydiff.^2)/length(Ydiff);
    Accuracy(i)=sum(Ydiff==0)/length(Ydiff);
    display("Iteration "+i+" rho="+rho+" MSE="+MSE(i)+" Accuracy="+Accuracy(i))
end

figure()
plot(Rho,MSE,'r',Rho,Accuracy,'b')
title("MSE & Accuracy vs Rho")
legend("MSE",'Accuracy')


