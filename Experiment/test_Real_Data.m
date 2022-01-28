%% find spatial and temporal parameters from data
clear all, close all, clc
figOpt = false;
printOpt = false;
[hyp_sp, stns_loc] = FitSpatialField(figOpt);
[hyp0, hyp1] = FitTemporalProcess(figOpt);

%% spatial field
meanfunc = @meanConst; 
covfunc = {@covMaterniso, 3};
c = hyp_sp.thres;

%% temporal processes 
% null hypotheis
meanfunc0 = @meanConst;  
covfunc0 = {@covMaterniso, 3};
pd0 = hyp0.dist;
% alternative hypothesis
meanfunc1 = @meanConst;  
covfunc1 = {@covMaterniso, 3};
pd1 = hyp1.dist;

% parameters for the sensor network, physical meaning is lost
T = 19 * 7; % time period for temporal processes [0, T]
M = 19 * 7; % number of point observations, take observation at the end of 
K = 19 * 7; % number of integral observations
snP = 0.1; % signal noise of point sensors
snI = 0.1; % signal noise of integral sensors
ratio = 1; % percentage of point sensors over all sensors
modelHyp = struct("T",T,"M",M,"K",K,"snI",snI,"snP",snP,'ratio',ratio);

% lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
warpdist0 = 'Gamma';warpdist1 = "Gamma";
[lb0,ub0] = lowUpBound(warpdist0,M);
[lb1,ub1] = lowUpBound(warpdist1,M);

% create structures to store hyper-parameters 
hyp0.lb = lb0;
hyp0.ub = ub0;

hyp1.lb = lb1;
hyp1.ub = ub1;

hypSp = struct("meanfunc", meanfunc, "covfunc", {covfunc}, "hyp", hyp_sp);
H0 = struct("meanfunc",meanfunc0,"covfunc",{covfunc0},"hyp",hyp0);
H1 = struct("meanfunc",meanfunc1,"covfunc",{covfunc1},"hyp",hyp1);

% define warping function
warpfunc = @(pd,p) invCdf(pd,p); % the inverseCDF warping function
warpinv = @(pd,p) invCdfWarp(pd,p); % inverse function of inverseCDF warping
warpfunc_sp = @(c,x) indicator(c,x); % the warping function of the binary spatial field is the indicator function

%% generate the synthetic spatial field
S = shaperead('SGP_adm0.shp'); % Singapore shape file
lnlim = [min(S.X) max(S.X)];
ltlim = [min(S.Y) max(S.Y)];
nx = 50;
ny = 50;

% generate grid point 
[x,y] = meshgrid(linspace(lnlim(1),lnlim(2),nx),...
    linspace(ltlim(1),ltlim(2),ny));

% append the sensor locations
xSp = [reshape(x,[],1); stns_loc(:,1)];
ySp = [reshape(y,[],1); stns_loc(:,2)]; 
X = [xSp,ySp];
% X = [X(:,1)-103, X(:,2)-1] * 10;
% generate the binary spatial random field
g = SimGP(hyp_sp,meanfunc,covfunc,X);
Y = warpfunc_sp(c,g);
% Y = SimWGP(hyp_sp,meanfunc,covfunc,warpfunc_sp,X);

%% generate point observations 
% assume all sensors are point sensors, i.e., integral sensors = []
% the tranining data are the observations from 21 sensors
% indexTrain = (nx * ny + 1 : size(X, 1))';
indexTrain = (nx * ny + 1 : size(X, 1))';
indexTest = setdiff(1:length(Y), indexTrain);

Xtrain=X(indexTrain,:); 
Xtest=X(indexTest,:);

Ytrain=Y(indexTrain);
Ytest=Y(indexTest);

% initialize an empty prediction array
Yhat=zeros(length(Y),1); 

% get locations of point senosrs with value of the spatial random field being 0
% and 1
Ntrain = length(indexTrain);
xP = indexTrain(1:Ntrain);
xP0= xP(Y(xP)==0);
xP1= xP(Y(xP)==1);
nP0= length(xP0);
nP1= length(xP1);

xI=setdiff(indexTrain,xP);
xI0=xI(Y(xI)==0);
xI1=xI(Y(xI)==1);
nI0=length(xI0);
nI1=length(xI1);

% generate the Point Observations
t=linspace(0,T,M)'; % the time points to observe the point observations

% parameters
CP0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
muP0 = meanfunc0( hyp0.mean, t);
CP1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
muP1 = meanfunc1( hyp1.mean, t);

ZP0 = SimPtData(hyp0,CP0,muP0,warpfunc,t,snP,nP0);
ZP1 = SimPtData(hyp1,CP1,muP1,warpfunc,t,snP,nP1);

%% WGPLRT
% run Laplace approximation
x_init = [ones(M,1)*pd0.mean, ones(M,1)*pd1.mean]; 
LRT = WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);

%%
% logGammaP at significance level alpha=0.1 from simulation
alpha = 0.1;
[wtp,wfp,logGammaP] = FuncWGPLRT(H0, H1, T, M, snP,alpha, printOpt,figOpt, LRT);
% predict the value of the spatial random field for sensors
yhat_pt_0 = WGPLRT_pred(ZP0,LRT,logGammaP); 
yhat_pt_1 = WGPLRT_pred(ZP1,LRT,logGammaP);

% assign predictions to the locations of point sensors
Yhat(xP0) = yhat_pt_0;
Yhat(xP1) = yhat_pt_1;

% collect the predictions 
Ytrain_hat  = Yhat(indexTrain);

%% SBLUE
% offline phase for SBLUE
SBLUEprep = SBLUE_stats_prep(covfunc,meanfunc,hyp_sp,Xtrain,Xtest); 
%%
% construct transition matrices for all sensors
liP = ismember(indexTrain,xP);  % the locations of the point observations (using WGPLRT)
liI = ismember(indexTrain,xI);  % the locations of the integral observations (using NLRT)
rho = [1-wfp,0.8972];lambda = [wtp,1]; % rho indicates the 1-FPR of WGPLRT & NLRT; lambda indicates TPR of WGPLRT & NLRT 
A1 = [rho(1),1-rho(1);1-lambda(1),lambda(1)];% transition matrix (WGPLRT)
A2 = [rho(2),1-rho(2);1-lambda(2),lambda(2)];% transition matrix (NLRT)
transitionMat = SBLUE_confusion(A1,A2,liP,liI);
% make predictions
SBLUE = SBLUE_stats(SBLUEprep,transitionMat,c); % calculate the SBLUE covariances 
Ypred=  SBLUE_pred(SBLUE,Ytrain_hat);   % predictions

%% evaluate the performance
MSE_SBLUE = sum((Ytest-Ypred).^2)/length(Ypred)
F1_SBLUE = F1score(Ytest,Ypred)

%% oracle
g_train = g(indexTrain);
KXX = feval(covfunc{:},hyp_sp.cov,Xtrain);
KxX = feval(covfunc{:}, hyp_sp.cov, Xtest, Xtrain);
m_test = meanfunc(hyp_sp.mean, Xtest);
g_pred= m_test + KxX / KXX * (g_train - hyp_sp.mean);
Ypred_GPR = double(g_pred > hyp_sp.thres);
MSE_GPR = sum((Ypred_GPR-Ytest).^2)/length(Ypred_GPR)
F1_GPR = F1score(Ytest,Ypred_GPR)

%% KNN
% Mdl = fitcknn(Xtrain,Ytrain_hat,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus',"ShowPlots",false));
%%
Mdl = fitcknn(Xtrain,Ytrain_hat,'Distance','minkowski','NumNeighbors',5);
[Ypred_KNN,score,cost] = predict(Mdl,Xtest);
MSE_KNN = sum((Ypred_KNN-Ytest).^2)/length(Ypred_KNN)
F1_KNN=F1score(Ytest,Ypred_KNN)

%% plot the heatmap
[fmask, vmask] = maskPatch(S);
z = reshape(Y(1:nx*ny), [ny, nx]);
figure()
mapshow(x, y, z, 'DisplayType','surface', ...
                 'zdata', ones(size(x))*0, ... % keep below gridlines
                 'cdata', z, ... 
                 'facecolor', 'flat');    
patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
axis equal
title("True binary spatial field")

zhat = reshape(Ypred,[ny,nx]);
figure()
mapshow(x, y, zhat, 'DisplayType','surface', ...
                 'zdata', ones(size(x))*0, ... % keep below gridlines
                 'cdata', zhat, ... 
                 'facecolor', 'flat');    
patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
axis equal
title("Reconstructed field")





