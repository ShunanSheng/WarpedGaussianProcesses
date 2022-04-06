function [MSE, F1, TPR, FPR]  = FuncSemiSyntheticExperiment(modelHyp, Options)
%% find spatial and temporal parameters from data
figOpt = Options.figOpt;
printOpt = Options.printOpt;

if exist('spatial_hyper.mat','file')
    hyp_sp = load('spatial_hyper.mat').hyp_final;
    stns_loc = load('spatial_hyper.mat').stns_loc;
else
    [hyp_sp, stns_loc] = FitSpatialField(figOpt);
end
if exist('temporal_hyper.mat','file')
    hyp0 = load('temporal_hyper.mat').hyp0;
    hyp1 = load('temporal_hyper.mat').hyp1;
else
    fprintf("No files found")
    [hyp0, hyp1] = FitTemporalProcess(figOpt);
end

fprintf("Load spatial and temporal hyperparameters successfully!\n")

%% spatial field
meanfunc = @meanConst; 
covfunc = {@covMaterniso, 5};
c = hyp_sp.thres;

%% temporal processes 
% null hypotheis
meanfunc0 = @meanConst;  
covfunc0 = {@covMaterniso, 5};
pd0 = hyp0.dist;
% alternative hypothesis
meanfunc1 = @meanConst;  
covfunc1 = {@covMaterniso, 5};
pd1 = hyp1.dist;

% parameters for the sensor network
T = modelHyp.T; % time period for temporal processes [0, T]
M = modelHyp.M; % number of point observations, take observation at the end of 
K = modelHyp.K; % number of integral observations
snP = modelHyp.snP; % signal noise of point sensors
snI = modelHyp.snI; % signal noise of integral sensors
ratio = modelHyp.ratio; % percentage of point sensors over all sensors
alpha = modelHyp.alpha; % the significance level for LRT

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

%% generate the synthetic spatial locations
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

%% partition the tranining and test index
% assume all sensors are point sensors, i.e., integral sensors = []
% the tranining data are the observations from the 21 sensors
indexTrain = (nx * ny + 1 : size(X, 1))';
indexTest = setdiff(1:size(X,1), indexTrain)';

% partition the training and test data
Xtrain=X(indexTrain,:); 
Xtest=X(indexTest,:);

%% generate the spatial field
% generate the binary spatial random field (with knowledge of the latent GP for GPR)
% rng("default")
g = SimGP(hyp_sp,meanfunc,covfunc,X);
Y = warpfunc_sp(c,g);

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
%% generate the Point Observations
t=linspace(0,T,M)'; % the time points to observe the point observations

% parameters
CP0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
muP0 = meanfunc0( hyp0.mean, t);
CP1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
muP1 = meanfunc1( hyp1.mean, t);

ZP0 = SimPtData(hyp0,CP0,muP0,warpfunc,t,snP,nP0);
ZP1 = SimPtData(hyp1,CP1,muP1,warpfunc,t,snP,nP1);

%% offline: get LRT and SBLUE prep
% load offline parameters if available
root = "Experiment/SemiSyntheticExperiment/Results/semisyn_offline_";
if Options.VaryParameter == 1
   file_name = strcat(root, 'M_',num2str(M),".mat");
elseif Options.VaryParameter == 2
   file_name = strcat(root, 'snP_',num2str(snP),".mat");
elseif Options.VaryParameter == 3
   file_name = strcat(root, "alpha_",num2str(alpha),".mat");
else
   error("Not recognized experiment type") 
end

if exist(file_name,'file')
%     fprintf("Load offline parameters !\n")
    LRT = load(file_name).LRT;
    wtp = load(file_name).wtp;  
    wfp = load(file_name).wfp;
    logGammaP = load(file_name).logGammaP;
    SBLUEprep = load(file_name).SBLUEprep;
    fprintf("Load offline parameters !\n")
else
    fprintf("Compute offline parameters !\n")
    % run Laplace approximation
    x_init = [ones(M,1)*pd0.mean, ones(M,1)*pd1.mean]; 
    LRT = WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);
    % logGammaP at significance level = alpha from simulation
    [wtp,wfp,logGammaP] = FuncWGPLRT(H0, H1, T, M, snP,alpha, printOpt,figOpt, LRT);
    % offline phase for SBLUE
    SBLUEprep = SBLUE_stats_prep(covfunc,meanfunc,hyp_sp,Xtrain,Xtest);
 
    % store the parameters
    save(file_name,'LRT','wtp','wfp','logGammaP','SBLUEprep');
end

fprintf("Start online computation !\n")
%% online: WGPLRT
% predict the value of the spatial random field for sensors
yhat_pt_0 = WGPLRT_pred(ZP0,LRT,logGammaP); 
yhat_pt_1 = WGPLRT_pred(ZP1,LRT,logGammaP);

% assign predictions to the locations of point sensors
Yhat(xP0) = yhat_pt_0;
Yhat(xP1) = yhat_pt_1;

% collect the predictions 
Ytrain_hat  = Yhat(indexTrain);
[TPR.Train, FPR.Train] = confusionMat(Ytrain_hat,Ytrain);
%% online : SBLUE
% construct transition matrices for all sensors
liP = ismember(indexTrain,xP);  % the locations of the point observations (using WGPLRT)
liI = ismember(indexTrain,xI);  % the locations of the integral observations (using NLRT)

% rho indicates the 1-FPR of WGPLRT & NLRT; lambda indicates TPR of WGPLRT & NLRT 
% here, we set TPR/FPR of NLRT to be 1 since we only have point
% observations
rho = [1-wfp,1];lambda = [wtp,1]; 
A1 = [rho(1),1-rho(1);1-lambda(1),lambda(1)];% transition matrix (WGPLRT)
A2 = [rho(2),1-rho(2);1-lambda(2),lambda(2)];% transition matrix (NLRT)
transitionMat = SBLUE_confusion(A1,A2,liP,liI); % the mixed transition matrix

% make predictions
SBLUE = SBLUE_stats(SBLUEprep,transitionMat,c); % calculate the SBLUE covariances 
[Ypred, g_star_pred] = SBLUE_pred(SBLUE,Ytrain_hat);   % predictions

%% evaluate the performance
MSE.SBLUE = sum((Ytest-Ypred).^2)/length(Ypred);
F1.SBLUE = F1score(Ytest,Ypred);
[TPR.SBLUE, FPR.SBLUE] = confusionMat(Ytest,Ypred);
%% Bayes risk
Cov_gstar = feval(covfunc{:}, hyp_sp.cov, Xtest);  % Cov(xtar, xstar)
Risk_SBLUE = SBLUE_risk(Cov_gstar, SBLUE);

%% oracle: GPR
g_train = g(indexTrain);
KXX = SBLUEprep.Cov_xtrain;
KxX = SBLUEprep.Cov_xstar;
m_test = meanfunc(hyp_sp.mean, Xtest);
g_pred= m_test + KxX / KXX * (g_train - hyp_sp.mean);
Ypred_GPR = double(g_pred > hyp_sp.thres);
MSE.GPR = sum((Ypred_GPR-Ytest).^2)/length(Ypred_GPR);
F1.GPR = F1score(Ytest,Ypred_GPR);
[TPR.GPR, FPR.GPR] = confusionMat(Ytest,Ypred_GPR);

%% KNN
% Mdl = fitcknn(Xtrain,Ytrain_hat,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus',"ShowPlots",false));
%%
Mdl = fitcknn(Xtrain,Ytrain_hat,'Distance','minkowski','NumNeighbors',5);
[Ypred_KNN,~,~] = predict(Mdl,Xtest);
MSE.KNN = sum((Ypred_KNN-Ytest).^2)/length(Ypred_KNN);
F1.KNN = F1score(Ytest,Ypred_KNN);
[TPR.KNN, FPR.KNN] = confusionMat(Ytest,Ypred_KNN);


%% plot the heatmap
if figOpt
    close all
    [fmask, vmask] = maskPatch(S);
    z_risk = reshape(Risk_SBLUE, [ny, nx]); 
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.065 .04],[.11 .01])
    mapshow(x, y, z_risk, 'DisplayType','surface', ...
                     'zdata', ones(size(x))*0, ... % keep below gridlines
                     'cdata', z_risk, ... 
                     'facecolor', 'flat');    
    patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
    axis equal
    hold on
    colorbar
    long = stns_loc(:,1);
    lat = stns_loc(:,2);
    text([long(:)],[lat(:)],'x','color','r','FontSize',15)
    xlabel("Longtitude",'FontSize',15)
    ylabel("Latitude",'FontSize',15)
    title("Bayes risk",'FontSize',20)
    
    z_true = reshape(g(1:nx*ny), [ny, nx]); 
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.065 .04],[.11 .01])
    mapshow(x, y, z_true, 'DisplayType','surface', ...
                     'zdata', ones(size(x))*0, ... % keep below gridlines
                     'cdata', z_true, ... 
                     'facecolor', 'flat');    
    patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
    axis equal
    hold on
    colorbar
    long = stns_loc(:,1);
    lat = stns_loc(:,2);
    text([long(:)],[lat(:)],'x','color','r','FontSize',15)
    xlabel("Longtitude",'FontSize',15)
    ylabel("Latitude",'FontSize',15)
    title("The True Gaussian Spatial Field",'FontSize',20)

    z_ghat = reshape(g_star_pred, [ny, nx]); 
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.065 .04],[.11 .01])
    mapshow(x, y, z_ghat, 'DisplayType','surface', ...
                     'zdata', ones(size(x))*0, ... % keep below gridlines
                     'cdata', z_ghat, ... 
                     'facecolor', 'flat');    
    patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
    axis equal
    hold on
    colorbar
    long = stns_loc(:,1);
    lat = stns_loc(:,2);
    text([long(:)],[lat(:)],'x','color','r','FontSize',15)
    xlabel("Longtitude",'FontSize',15)
    ylabel("Latitude",'FontSize',15)
    title("The Reconstructed Gaussian Spatial Field",'FontSize',20)
    
     
    z = reshape(Y(1:nx*ny), [ny, nx]);
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.065 .04],[.11 .01])
    mapshow(x, y, z, 'DisplayType','surface', ...
                     'zdata', ones(size(x))*0, ... % keep below gridlines
                     'cdata', z, ... 
                     'facecolor', 'flat');    
    patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
    axis equal
    hold on
    colorbar
    long = stns_loc(:,1);
    lat = stns_loc(:,2);
    text([long(:)],[lat(:)],'x','color','r','FontSize',15)
    xlabel("Longtitude",'FontSize',15)
    ylabel("Latitude",'FontSize',15)
    title("The True Binary Spatial Field",'FontSize',20)

    zhat = reshape(Ypred,[ny,nx]);
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.065 .04],[.11 .01])
    mapshow(x, y, zhat, 'DisplayType','surface', ...
                     'zdata', ones(size(x))*0, ... % keep below gridlines
                     'cdata', zhat, ... 
                     'facecolor', 'flat');    
    patch('Faces',fmask,'Vertices',vmask,'FaceColor','w','EdgeColor','none');
    axis equal
    long = stns_loc(:,1);
    lat = stns_loc(:,2);
    colorbar
    text([long(:)],[lat(:)],'x','color','r','FontSize',15)
    xlabel("Longtitude",'FontSize',15)
    ylabel("Latitude",'FontSize',15)
    title("The True Binary Spatial Field",'FontSize',20)
    title("The Reconstructed Binary Spatial Field")
end
end

