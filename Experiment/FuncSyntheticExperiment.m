function [F1_SBLUE,TPR,FPR]=FuncSyntheticExperiment(M,sn,alpha,ratio,printOpt,figOpt)
% Compute the F1 score given the varying parameters M, sn

% Setup for Spatial Random field
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp.cov=log([ell; sf]); 
c=0;pd=[];% that is q=0.5
hyp=struct('mean',0,'cov',hyp.cov,'dist',pd,'thres',c);


%%% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covMaterniso, 1}; ell1=1; sf1=1; hyp0.cov=log([ell1; sf1]);
pd0 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

%%% H1 Alternative hypothesis

meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 5}; ell1=1; sf1=1; hyp1.cov=log([ell1; sf1]);
pd1 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);


%%% Parameters for the sensor network
T=20;K=M;snP=sn;snI=sn;
modelHyp=struct("T",T,"M",M,"K",K,"snI",snI,"snP",snP,'ratio',ratio);

%%% Lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
warpdist0="Normal";warpdist1="Normal";

[lb0,ub0]=lowUpBound(warpdist0,M);
[lb1,ub1]=lowUpBound(warpdist1,M);

%%% Create structures to store the hyperparameters
SP=struct("meanfunc",meanfunc,"covfunc",covfunc,"hyp",hyp);
hyp0=struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1=struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0=struct("meanfunc",meanfunc0,"covfunc",covfunc0,"hyp",hyp0);
H1=struct("meanfunc",meanfunc1,"covfunc",covfunc1,"hyp",hyp1);

%%% Warping function
warpfunc=@(pd,p) invCdf(pd,p);
warpinv=@(pd,p) invCdfWarp(pd,p);
warpfunc_sf=@(c,x) indicator(c,x); % the warping function of the binary spatial field is the indicator function

%%% Generate synthetic data
Data=SimSynData(SP,H0,H1,warpfunc_sf, warpfunc, modelHyp);

%%% Extract information from Data, e.g. sensor locations, latent labels
x=Data.x;
y=Data.y;
indexTrain=Data.indexTrain; % index of tranining locations
indexTest=Data.indexTest; % index of test locations

Xtrain=x(indexTrain,:); 
Xtest=x(indexTest,:);

Ytrain=y(indexTrain);
Ytest=y(indexTest);

%% Offline Phase
if printOpt==true
disp("Offline phase!")
end
%% Offline: WGPLRT
% disp("Offline: WGPLRT")
t=Data.t;ZP0=Data.ZP.H0;ZP1=Data.ZP.H1;xP0=Data.xP.H0;xP1=Data.xP.H1;

% run Laplace approximation
x_init=[ones(M,1)*0.5, ones(M,1)*0.5]; 
LRT=WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);

[wtp,wfp,logGammaP]=FuncSimWGPLRT(M,sn,alpha,false,false);
% logGammaP=203.5655; %logGammaP at significance level alpha=0.1 from simulation


%% Offline: NLRT
% disp("Offline: NLRT")
% The Integral Observations
ZI0=Data.ZI.H0;
ZI1=Data.ZI.H1;
xI0=Data.xI.H0;
xI1=Data.xI.H1;

kw= ceil(exp(log(10000*T/K/180)/4)); % calculate the number of point neeed per window under Simpson's rule with 0.01 error
kw= round(kw/2)*2;n=kw*K;tI=linspace(0,T,n)';

CI0 = chol(feval(covfunc0{:}, hyp0.cov, tI)+1e-9*eye(n));
muI0 = meanfunc0( hyp0.mean, tI);

CI1 = chol(feval(covfunc1{:}, hyp1.cov, tI)+1e-9*eye(n));
muI1 = meanfunc1( hyp1.mean, tI);

% Generate J samples of integral observations from null and alternative hypotheses
sumstats=@(z) summaryAuto(z,4); % the summary statistic: Autocorrelation with lag=1,...,4
d=@distEuclid; % distance measure: Euclidean metric
J=100000; % number of generated samples per hypothesis
[Z0,Z1]=NLRT_gene(hyp0,CI0,muI0,hyp1,CI1,muI1, warpfunc,K,kw,snI,J); 

% Parameters for NLRT
delta=0.1; % distance tolerance
[ntp,nfp,logGammaI]=FuncSimNLRT(M,sn,alpha,false,false);
% logGammaI=-0.5182; %logGammaI at significance level alpha=0.1 from simulation


%% Offline: SBLUE
% disp("Offline: SBLUE")
SBLUEprep=SBLUE_stats_prep(covfunc,meanfunc,hyp,Xtrain,Xtest); 
if printOpt==true
disp("Offline phase finished!")
end

%% Online phase
if printOpt==true
disp("Online Phase")
end
Yhat=zeros(length(y),1);   % the vector to store the decisions from LRTs

%% Online: WGPLRT
% disp("Online: WGPLRT")
% LRT for ZP0 and ZP1
yhat_pt_0=WGPLRT_pred(ZP0,LRT,logGammaP); % the classification
yhat_pt_1=WGPLRT_pred(ZP1,LRT,logGammaP);% the classification

% Assign predictions to the locations of point observations
Yhat(xP0)=yhat_pt_0;
Yhat(xP1)=yhat_pt_1;
%% Online: NLRT
% disp("Online: NLRT")
% NLRT for Z0
[D00,D01]=NLRT_stats(ZI0,Z0,Z1,sumstats,d); % compute the distance matrix
Lambda0=NLRT_pred_delta(D00,D01,delta);
yhat_int_0=NLRT_pred_gamma(Lambda0,logGammaI); 

% NLRT for Z1
[D10,D11]=NLRT_stats(ZI1,Z0,Z1,sumstats,d);
Lambda1=NLRT_pred_delta(D10,D11,delta);
yhat_int_1=NLRT_pred_gamma(Lambda1,logGammaI); 

% Assign predictions to the locations of integral observations
Yhat(xI0)=yhat_int_0;
Yhat(xI1)=yhat_int_1;

Ytrain_hat=Yhat(indexTrain);

%% Online: SBLUE
% disp("Online: SBLUE")
% the false postive rates and true postive rates are
% computed via simulation in test_WGPLRT.m and test_NLRT.m
tic
liP=ismember(indexTrain,[xP0;xP1]);        % the locations of the point observations (using WGPLRT)
liI=ismember(indexTrain,[xI0;xI1]);        % the locations of the integral observations (using NLRT)
rho=[1-wfp,1-nfp];lambda=[wtp,ntp]; % rho indicates the 1-FPR of WGPLRT & NLRT; lambda indicates TPR of WGPLRT & NLRT 
A1=[rho(1),1-rho(1);1-lambda(1),lambda(1)];% transition matrix (WGPLRT)
A2=[rho(2),1-rho(2);1-lambda(2),lambda(2)];% transition matrix (NLRT)

transitionMat=SBLUE_confusion(A1,A2,liP,liI);
SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c); % calculate the SBLUE covariances 
Ypred=SBLUE_pred(SBLUE,Ytrain_hat);           % predictions
F1_SBLUE=F1score(Ytest,Ypred);
[TPR,FPR]=confusionMat(Ytest,Ypred);
t_SBLUE=toc;

%% Evaluation
if printOpt==true
clc
% Overall, training loss
[tp,fp]=confusionMat(Ytrain,Ytrain_hat);

% WGPLRT
YP_hat=[yhat_pt_0;yhat_pt_1];
YP=[y(xP0);y(xP1)];
[wtp,wfp]=confusionMat(YP,YP_hat);

% NLRT
YI_hat=[yhat_int_0;yhat_int_1];
YI=[y(xI0);y(xI1)];
[ntp,nfp]=confusionMat(YI,YI_hat);

fprintf("Overall :TPR= %4.3f, FPR=%4.3f, MSE=%4.3f\n",tp, fp, sum((Ytrain-Ytrain_hat).^2)/length(Ytrain));
fprintf("WGPLRT :TPR= %4.3f, FPR=%4.3f, MSE=%4.3f\n",wtp, wfp, sum((YP-YP_hat).^2)/length(YP));
fprintf("NLRT :TPR= %4.3f, FPR=%4.3f, MSE=%4.3f\n",ntp, nfp, sum((YI-YI_hat).^2)/length(YI));
fprintf('SBLUE w noise has F1 score=%4.3f with time= %4.3f\n',F1_SBLUE,t_SBLUE);
% fprintf('KNN w noise has F1 score= %4.3f with time= %4.3f\n',F1_KNN,t_KNN);
end
%% Plot the graph
if figOpt==true
    close all 
    % Plot the latent field
    X=reshape(x(:,1),[50,50]);
    Y=reshape(x(:,2),[50,50]);
    Z=double(reshape(y,[50,50]));
    figure()
    surf(X,Y,Z,"DisplayName","latent binary field");
    shading interp
    view(2)
    colorbar
    hold on
    plot3(Xtest(:,1),Xtest(:,2),Ytest, ['x','r'],'DisplayName','un-monitored location');
    legend()
    hold off
    title('True Latent Binary Spatial Field')

    % Plot the prediction
    Yhat(indexTest)=Ypred;
    Zhat=double(reshape(Yhat,[50,50]));
    figure()
    surf(X,Y,Zhat,"DisplayName","latent binary field");
    shading interp
    view(2)
    colorbar
    hold on
    plot3(Xtest(:,1),Xtest(:,2),Ypred, ['x','r'],'DisplayName','un-monitored location');
    legend()
    title("The Reconstructed Binary Spatial Field")
end

end