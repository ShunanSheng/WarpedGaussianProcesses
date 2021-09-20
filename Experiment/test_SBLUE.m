%%% Test of SBLUE on 1D and 2D simulated data
clear all;close all;clc;

% Set up the spatial fied
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1; sf = 1; hyp.cov=log([ell; sf]);
q=0.5;
pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd);
warpfunc=@(pd,p) invCdf(pd,p);


%% 1D data
% N = 1000;
% x = linspace(-10,10,N)'; % Location of sensors
% % Simulate Warped Gaussian Process
% z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);

% %% 2D data
% % Location of sensors 
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

%% Use different A1 and A2 for SBLUE
clc;
rho=[0.9,0.99];delta=[0.9,0.99];
A1=[rho(1),1-rho(1);1-delta(1),delta(1)];
A2=[rho(2),1-rho(2);1-delta(2),delta(2)];
xP=indexTrain(1:2:end);
xI=setdiff(indexTrain,xP);
liP=ismember(indexTrain,xP)';
liI=ismember(indexTrain,xI)';

M=1000;
YT=repmat(Ytrue,[1,M]);
Yhat_noise=repmat( Yhat, [1,M] );
for j=1:M
    rnd=rand(length(Yhat),1);
    rnd1=rnd(liP)>rho(1);
    rnd2=rnd(liI)>rho(2);
    
    Yhat_noise(liP,j)=(1-rnd1).*Yhat_noise(liP,j)...
                                +rnd1.*(1-Yhat_noise(liP,j));
                            
    Yhat_noise(liI,j)=(1-rnd2).*Yhat_noise(liI,j)...
                                +rnd2.*(1-Yhat_noise(liI,j));                        
end


% Apply SBLUE
SBLUE=SBLUE_stats(SBLUEprep,A1,A2,liP,liI,q);
Ypred=SBLUE_pred(SBLUE,Yhat_noise);
% Evaluate the MSE and Accuracy
MSE=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));



%%
clc;
% In order to plot MSE graph to see the change of MSE over Rho
% we may use Rho to control both true negative & true positive rate for A1,A2
Rho=linspace(0,1,100)';
L=length(Rho);MSE=zeros(L,1);

M=10000;
YT=repmat(Ytrue,[1,M]);
for i=1:L % We expect when rho is increasing, the performance will become better
    rho=Rho(i);delta=Rho(i);
    A=[rho,1-rho;1-delta,delta]; % Define the transition matirx
    % Simulate the noisy data
    Yhat_noise=repmat( Yhat, [1,M] );
    for j=1:length(Yhat_noise(:))
        if rand()>rho
            Yhat_noise(j)=1-Yhat_noise(j);
        end
    end
    % Apply SBLUE
    SBLUE=SBLUE_stats(SBLUEprep,A1,A2,liP,liI,q);
    Ypred=SBLUE_pred(SBLUE,Yhat_noise);
    % Evaluate the MSE and Accuracy
    MSE(i)=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));
    if mod(i,floor(L/10))==0
        disp("Iteration "+i+",rho="+rho+", MSE="+MSE(i))
    end
end


%%
close all;
figure();
plot(Rho,MSE,"DisplayName","MSE");
title("MSE vs Rho")
xlabel("Rho")
ylabel("MSE")
legend("MSE")
hold on;

