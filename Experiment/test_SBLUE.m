%%% Test of SBLUE on 1D and 2D simulated data
clear all;close all;clc;
% Set up the spatial fied
meanfunc = @meanConst;hyp.mean=0;
% covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp.cov=log([ell; sf]);
% covfunc={@covFBM};sf0=1;h0=1/2;hyp.cov=[log(sf0);-log(1/h0-1)];
% covfunc = {@covMaterniso, 3}; ell=2; sf=1; hyp.cov=log([ell; sf]);
covfunc = {@covMaterniso, 3}; ell=exp(1); sf=1; hyp.cov=log([ell; sf]);
% q=0.5;pd=makedist("Binomial",'N',1,'p',q); % Bernouli(p)
pd=[];c=0;
hyp=struct('mean',hyp.mean,'cov',hyp.cov,'dist',pd,'thres',c);
% warpfunc=@(pd,p) invCdf(pd,p);
warpfunc=@(c,x) indicator(c,x);
rng("default")

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
clc;
indexTest=1:5:N;
indexTrain=setdiff(1:N,indexTest);

Ytrain=z(indexTrain);
Ytest=z(indexTest);
Xtrain=x(indexTrain,:);
Xtest=x(indexTest,:);

%% SBLUE prep
SBLUEprep=SBLUE_stats_prep(covfunc,meanfunc,hyp,Xtrain,Xtest); 

%% pure SBLUE without noise
tic;
rho=[1,1];lambda=[1,1];
A1=[rho(1),1-rho(1);1-lambda(1),lambda(1)];
A2=[rho(2),1-rho(2);1-lambda(2),lambda(2)];
xP=indexTrain(1:2:end);
xI=setdiff(indexTrain,xP);
liP=ismember(indexTrain,xP)';
liI=ismember(indexTrain,xI)';

M=1000;
YT=repmat(Ytest,[1,M]);
Yhat_noise=repmat( Ytrain, [1,M] );
% Generate the noisy data
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
% Compute the adjusted confusion probability

% Apply SBLUE
SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c);
Ypred=SBLUE_pred(SBLUE,Yhat_noise);
% Evaluate the MSE and Accuracy
MSE_SBLUE=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));
t_SBLUE=toc;

%% noisy SBLUE, assume p00=p11 in this case
tic
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
t_SBLUE_noisy=toc;


%% GPR without noise i.e. g_star=K(xstar,Xtrain)K(Xtrain,Xtrain)^{-1}g
tic;
g_train=g(indexTrain);
KXX=feval(covfunc{:},hyp.cov,Xtrain);
KxX=feval(covfunc{:}, hyp.cov, Xtest, Xtrain);
g_pred=KxX / KXX * g_train;
Ypred=g_pred>c;
MSE_GPR=sum((Ypred-Ytest).^2)/length(Ypred)
t_GPR=toc;

%% KNN, run KNN using the noisy data
tic
XTrain=repmat(Xtrain,[M,1]);XTest=repmat(Xtest,[M,1]);
Mdl = fitcknn(XTrain,Yhat_noise(:),'NumNeighbors',5,'Standardize',1);
[Ypred,score,cost] = predict(Mdl,XTest);
MSE_KNN_noisy=sum((Ypred-YT(:)).^2)/length(YT(:));
t_KNN_noisy=toc;
%% KNN, wo noise
tic;
Mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'Standardize',1);
[Ypred,score,cost] = predict(Mdl,Xtest);
MSE_KNN=sum((Ypred-Ytest).^2)/length(Ypred);
t_KNN=toc;

fprintf('SBLUE w noise has MSE= %4.3f, with time= %4.3f\n',MSE_SBLUE_noisy,t_SBLUE_noisy);
fprintf('KNN w noise has MSE= %4.3f,with time= %4.3f\n',MSE_KNN_noisy,t_KNN_noisy);
fprintf('SBLUE wo noise has MSE= %4.3f,with time= %4.3f\n',MSE_SBLUE,t_SBLUE);
fprintf('KNN w0 noise has MSE= %4.3f,with time= %4.3f\n',MSE_KNN,t_KNN);
fprintf('GPR wo noise has MSE= %4.3f,with time= %4.3f\n',MSE_GPR,t_GPR);


% In order to plot MSE graph to see the change of MSE over Rho
% we may use Rho to control both true negative & true positive rate for A1,A2
% Rho=linspace(0.01,1,100)'; 
% FPR=linspace(0.01,1,100)';k=10;beta=(1+exp(-k))/(1-exp(-k));alpha=2*beta;
% TPR=alpha./(1+exp(-k.*FPR))-beta;% create TPR/FPR vector

%%
TP=load('TP.mat','TP');TPR=TP.TP;
FP=load('FP.mat','FP');FPR=FP.FP;
TPR=TPR(336:1500);
FPR=FPR(336:1500);
%%

L=length(FPR);MSE=zeros(L,1);

M=1000;
YT=repmat(Ytest,[1,M]);
for i=1:L % We expect when rho is extreme, i.e. close to 0 or 1
    % the performance is better since it gives more information to the
    % SBLUE
    tp=TPR(i);fp=FPR(i); % Suppose ROC : y=x 
    A=[1-fp,fp;1-tp,tp]; % Define the confusion matrix for both A1,A2
    % Simulate the noisy data
    Yhat_noise=repmat( Ytrain, [1,M] );
    for j=1:M
        rnd=rand(length(Ytrain),1);
        rnd1=rnd>tp; % filp the 1 to 0 with prob 1-tp
        rnd2=rnd> (1-fp);% filp the 0 to 1 with prob fp
 
        Y1=Yhat_noise(:,j)==1;
        Y0=Yhat_noise(:,j)==0;
        
        idP1=logical(liP.*Y1);
        idP0=logical(liP.*Y0);
        
        idI1=logical(liI.*Y1);
        idI0=logical(liI.*Y0);
        
        Yhat_noise(idP1,j)=(1-rnd1(idP1)).*Yhat_noise(idP1,j)...
                                    +rnd1(idP1).*(1-Yhat_noise(idP1,j));
        Yhat_noise(idP0,j)=(1-rnd2(idP0)).*Yhat_noise(idP0,j)...
                                    +rnd2(idP0).*(1-Yhat_noise(idP0,j));
                                    
        Yhat_noise(idI1,j)=(1-rnd1(idI1)).*Yhat_noise(idI1,j)...
                                    +rnd1(idI1).*(1-Yhat_noise(idI1,j));
        Yhat_noise(idI0,j)=(1-rnd2(idI0)).*Yhat_noise(idI0,j)...
                                    +rnd2(idI0).*(1-Yhat_noise(idI0,j)); 
    end
    
    transitionMat=SBLUE_confusion(A,A,liP,liI);
    % Apply SBLUE
    SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c);
    Ypred=SBLUE_pred(SBLUE,Yhat_noise);
    % Evaluate the MSE and Accuracy
    MSE(i)=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));
    if mod(i,floor(L/10))==0
        fprintf("Iteration %d, fp=%4.2f, MSE=%4.2f\n",i,fp, MSE(i));
    end
end


%%
close all;
figure();
plot(FPR,MSE,"DisplayName","MSE");
hold on 
plot(FPR, TPR)
title("MSE vs FPR")
xlabel("FPR")
ylabel("MSE")
legend("MSE","TPR")
hold on;
