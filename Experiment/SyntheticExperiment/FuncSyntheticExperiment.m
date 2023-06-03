function [MSE, F1, TPR, FPR] = FuncSyntheticExperiment(modelHyp, Options)

figOpt = Options.figOpt;
printOpt = Options.printOpt;
% Setup for Spatial Random field
meanfunc = @meanConst; 
covfunc = {@covSEiso}; ell = 1/2; sf = 1; hyp_sp.cov = log([ell; sf]); 
c = 0;pd = [];% that is q=0.5
hyp_sp = struct('mean',0,'cov',hyp_sp.cov,'dist',pd,'thres',c);

% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covMaterniso, 1}; ell0 = 1; sf0 = 1; hyp0.cov = log([ell0; sf0]);
pd0 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

% H1 Alternative hypothesis
meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 5}; ell1 = 1; sf1 = 1; hyp1.cov = log([ell1; sf1]);
pd1 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

% Parameters for the sensor network
T = modelHyp.T; % time period for temporal processes [0, T]
M = modelHyp.M; % number of point observations, take observation at the end of 
K = modelHyp.K; % number of integral observations
snP = modelHyp.snP; % signal noise of point sensors
snI = modelHyp.snI; % signal noise of integral sensors
ratio = modelHyp.ratio; % percentage of point sensors over all sensors
alpha = modelHyp.alpha; % the significance level for LRT

% Lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
warpdist0 = "Normal";
warpdist1 = "Normal";

[lb0,ub0] = lowUpBound(warpdist0,M);
[lb1,ub1] = lowUpBound(warpdist1,M);

% Create structures to store the hyperparameters
hypSp = struct("meanfunc",meanfunc,"covfunc",{covfunc},"hyp",hyp_sp);
hyp0 = struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1 = struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0 = struct("meanfunc",meanfunc0,"covfunc",{covfunc0},"hyp",hyp0);
H1 = struct("meanfunc",meanfunc1,"covfunc",{covfunc1},"hyp",hyp1);

% Warping function
warpfunc = @(pd,p) invCdf(pd,p);
warpinv = @(pd,p) invCdfWarp(pd,p);
warpfunc_sp = @(c,x) indicator(c,x); % the warping function of the binary spatial field is the indicator function

if exist('Experiment/SyntheticExperiment/Results/SpatialLocation.mat','file')
    file = load('Experiment/SyntheticExperiment/Results/SpatialLocation.mat');
    x = file.x;
    indexTrain = file.indexTrain;
    indexTest = file.indexTest;
else 
    rng('default')
    % Generate synthetic data   
    n = 50; xinf = -5; xsup = 5;
    N = n.^2;
    [X,Y] = meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
    xSp = reshape(X,[],1);
    ySp = reshape(Y,[],1); 
    x = [xSp,ySp];

    % The index of training and test data
    indexTrain = randperm(N,floor(N/10))'; % index of training locations
    indexTest = setdiff(1:N,indexTrain)'; % index of test locations
    save('Experiment/SyntheticExperiment/Results/SpatialLocation.mat','x','indexTrain','indexTest');
end

locHyp = struct('x',x,'indexTrain',indexTrain, 'indexTest',indexTest);
Data = SimSynData(hypSp,H0,H1,warpfunc_sp, warpfunc, modelHyp, locHyp);

% Extract information from Data, e.g. sensor locations, latent labels
g = Data.g;
y = Data.y;

% Partition tranining and test sets
Xtrain = x(indexTrain,:); 
Xtest = x(indexTest,:);

Ytrain = y(indexTrain);
Ytest = y(indexTest);

% Point Observations
t = Data.t;
ZP0 = Data.ZP.H0;
ZP1 = Data.ZP.H1; 
xP0 = Data.xP.H0; 
xP1 = Data.xP.H1;

% Integral Observations
ZI0 = Data.ZI.H0;
ZI1 = Data.ZI.H1;
xI0 = Data.xI.H0;
xI1 = Data.xI.H1;

%% Offline phase
root = "Experiment/SyntheticExperiment/Results/syn_offline_";
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
    fprintf("Load offline parameters when M = %d, snP = %4.2f, alpha = %4.2f!\n", M, snP, alpha);
    file = load(file_name);
    LRT = file.LRT;
    wtp = file.wtp;  
    wfp = file.wfp;
    logGammaP = file.logGammaP;
    
    Z0 = file.Z0;
    Z1 = file.Z1;  
    ntp = file.ntp;
    nfp = file.nfp;
    logGammaI = file.logGammaI;
    
    SBLUEprep = file.SBLUEprep;
else
    fprintf("Compute offline parameters when M = %d, snP = %4.2f, alpha = %4.2f!\n", M, snP, alpha);
    % WGPLRT
    % run Laplace approximation
    x_init = [ones(M,1)*pd0.mean, ones(M,1)*pd1.mean]; 
    LRT = WGPLRT_opt(H0,H1,warpinv,t,x_init, snP);
    % logGammaP at significance level = alpha from simulation
    [wtp,wfp,logGammaP] = FuncWGPLRT(H0, H1, T, M, snP,alpha, printOpt,figOpt, LRT);
    
    % NLRT
    % calculate the number of point neeed per window under Simpson's rule with 0.01 error
    kw = ceil(exp(log(1000000*(T/K).^5/180)/4)); 
    kw = round(kw/2)*2;
    if kw < 4
        kw = 4; % at least four points in an integration window
    end
    n = kw * K;
    tI = linspace(0,T,n)';

    CI0 = chol(feval(covfunc0{:}, hyp0.cov, tI)+1e-9*eye(n));
    muI0 = meanfunc0( hyp0.mean, tI);

    CI1 = chol(feval(covfunc1{:}, hyp1.cov, tI)+1e-9*eye(n));
    muI1 = meanfunc1( hyp1.mean, tI);

    % Generate J samples of integral observations from null and alternative hypotheses
    J = 10000; % number of generated samples per hypothesis
    [Z0,Z1] = NLRT_gene(hyp0,CI0,muI0,hyp1,CI1,muI1, warpfunc,K,kw,snI,J); 
    
    % Parameters for NLRT
    [ntp,nfp,logGammaI] = FuncNLRT(H0, H1, T, K, snI, alpha, printOpt, figOpt, Z0, Z1);
    
    % SBLUE
    SBLUEprep = SBLUE_stats_prep(covfunc,meanfunc,hyp_sp,Xtrain,Xtest);
    % store the parameters
    save(file_name,'LRT','wtp','wfp','logGammaP','ntp', 'nfp',...
        'logGammaI','SBLUEprep','Z0', 'Z1');
end


%% Online phase
fprintf("Online computation!\n")
Yhat = zeros(length(y),1);   % the vector to store the decisions from LRTs

%% Online: WGPLRT
% LRT for ZP0 and ZP1
yhat_pt_0 = WGPLRT_pred(ZP0,LRT,logGammaP); % the classification
yhat_pt_1 = WGPLRT_pred(ZP1,LRT,logGammaP);% the classification

% Assign predictions to the locations of point observations
Yhat(xP0) = yhat_pt_0;
Yhat(xP1) = yhat_pt_1;
%% Online: NLRT
sumstats = @(z) summaryAuto(z,4); % the summary statistic: Autocorrelation with lag=1,...,4
d = @distEuclid; % distance measure: Euclidean metric
delta = 0.1; % distance tolerance
% NLRT for Z0
[D00,D01] = NLRT_stats(ZI0,Z0,Z1,sumstats,d); % compute the distance matrix
Lambda0 = NLRT_pred_delta(D00,D01,delta);
yhat_int_0 = NLRT_pred_gamma(Lambda0,logGammaI); 

% NLRT for Z1
[D10,D11] = NLRT_stats(ZI1,Z0,Z1,sumstats,d);
Lambda1 = NLRT_pred_delta(D10,D11,delta);
yhat_int_1 = NLRT_pred_gamma(Lambda1,logGammaI); 

% Assign predictions to the locations of integral observations
Yhat(xI0) = yhat_int_0;
Yhat(xI1) = yhat_int_1;

Ytrain_hat = Yhat(indexTrain);

%% Online: SBLUE
% the false postive rates and true postive rates are
% computed via simulation in test_WGPLRT.m and test_NLRT.m
% tic
liP = ismember(indexTrain,[xP0;xP1]);        % the locations of the point observations (using WGPLRT)
liI = ismember(indexTrain,[xI0;xI1]);        % the locations of the integral observations (using NLRT)
rho = [1-wfp,1-nfp];lambda = [wtp,ntp];        % rho indicates the 1-FPR of WGPLRT & NLRT; lambda indicates TPR of WGPLRT & NLRT 
A1 = [rho(1),1-rho(1);1-lambda(1),lambda(1)];% transition matrix (WGPLRT)
A2 = [rho(2),1-rho(2);1-lambda(2),lambda(2)];% transition matrix (NLRT)

transitionMat = SBLUE_confusion(A1,A2,liP,liI);
SBLUE = SBLUE_stats(SBLUEprep,transitionMat,c); % calculate the SBLUE covariances 
Ypred_SBLUE = SBLUE_pred(SBLUE,Ytrain_hat);           % predictions
F1.SBLUE = F1score(Ytest,Ypred_SBLUE);
MSE.SBLUE = sum((Ytest-Ypred_SBLUE).^2)/length(Ytest);
[TPR.SBLUE,FPR.SBLUE] = confusionMat(Ytest,Ypred_SBLUE);

%% SBLUE with point observations only
SBLUEprep_pobs = SBLUE_stats_prep(covfunc,meanfunc,hyp_sp,Xtrain(liP,:),Xtest);
transitionMat_pobs.p01 = A1(3);
transitionMat_pobs.p11 = A1(4);
SBLUE_pobs = SBLUE_stats(SBLUEprep_pobs,transitionMat_pobs,c); % calculate the SBLUE covariances 
Ypred_SBLUE_pobs = SBLUE_pred(SBLUE_pobs,Ytrain_hat(liP));% predictions
F1.SBLUE_pobs = F1score(Ytest,Ypred_SBLUE_pobs);
MSE.SBLUE_pobs = sum((Ytest-Ypred_SBLUE_pobs).^2)/length(Ytest);
[TPR.SBLUE_pobs,FPR.SBLUE_opbs] = confusionMat(Ytest,Ypred_SBLUE_pobs);

%% SBLUE with integral observations only
SBLUEprep_iobs = SBLUE_stats_prep(covfunc,meanfunc,hyp_sp,Xtrain(liI,:),Xtest);
transitionMat_iobs.p01 = A2(3);
transitionMat_iobs.p11 = A2(4);
SBLUE_iobs = SBLUE_stats(SBLUEprep_iobs,transitionMat_iobs,c); % calculate the SBLUE covariances 
Ypred_SBLUE_iobs = SBLUE_pred(SBLUE_iobs,Ytrain_hat(liI));% predictions
F1.SBLUE_iobs = F1score(Ytest,Ypred_SBLUE_iobs);
MSE.SBLUE_iobs = sum((Ytest-Ypred_SBLUE_iobs).^2)/length(Ytest);
[TPR.SBLUE_iobs,FPR.SBLUE_oibs] = confusionMat(Ytest,Ypred_SBLUE_iobs);

%% GPR: Oracle
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
% tic
Mdl = fitcknn(Xtrain,Ytrain_hat,'Distance','seuclidean','NumNeighbors',7);
[Ypred_KNN, ~ , ~] = predict(Mdl,Xtest);
F1.KNN = F1score(Ytest,Ypred_KNN);
MSE.KNN=sum((Ypred_KNN-Ytest).^2)/length(Ytest);
[TPR.KNN,FPR.KNN] = confusionMat(Ytest,Ypred_KNN);
% time.KNN = toc;

%% Evaluation
if printOpt
% Overall, training loss
[tp,fp] = confusionMat(Ytrain,Ytrain_hat);

% WGPLRT
YP_hat = [yhat_pt_0;yhat_pt_1];
YP = [y(xP0);y(xP1)];
[wtp,wfp] = confusionMat(YP,YP_hat);

% NLRT
YI_hat = [yhat_int_0;yhat_int_1];
YI = [y(xI0);y(xI1)];
[ntp,nfp] = confusionMat(YI,YI_hat);

fprintf("Overall :TPR= %4.3f, FPR=%4.3f, MSE=%4.3f\n",tp, fp, sum((Ytrain-Ytrain_hat).^2)/length(Ytrain));
fprintf("WGPLRT :TPR= %4.3f, FPR=%4.3f, MSE=%4.3f\n",wtp, wfp, sum((YP-YP_hat).^2)/length(YP));
fprintf("NLRT :TPR= %4.3f, FPR=%4.3f, MSE=%4.3f\n",ntp, nfp, sum((YI-YI_hat).^2)/length(YI));
fprintf('SBLUE w noise has F1 score=%4.3f with MSE= %4.3f\n',F1.SBLUE,MSE.SBLUE);
fprintf('KNN w noise has F1 score= %4.3f with MSE= %4.3f\n',F1.KNN,MSE.KNN);
end
%% Plot the graph
if figOpt
    close all 
    % Plot the latent field
    X = reshape(x(:,1),[50,50]);
    Y = reshape(x(:,2),[50,50]);
    Z = double(reshape(y,[50,50]));
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.05 .075],[.05 .01])
    surf(X,Y,Z);
    shading interp
    view(2)
    colorbar
    hold on
    plot3(Xtrain(:,1),Xtrain(:,2),Ytrain, ['x','r']);
    legend(["true binary spatial field",'sensor locations'],'Location','southeast','FontSize',15)
    hold off
    title('The True Binary Spatial Field','FontSize',20)
    savefig("Experiment/SyntheticExperiment/Figs/TrueField.fig")

    % Plot the prediction
    Yhat(indexTest) = Ypred_SBLUE;
    Zhat = double(reshape(Yhat,[50,50]));
    figure('Position',[100,100,400,300])
    tight_subplot(1,1,[.01 .03],[.05 .075],[.05 .01])
    surf(X,Y,Zhat);
    shading interp
    view(2)
    colorbar
    hold on
    plot3(Xtrain(:,1),Xtrain(:,2),Ytrain_hat, ['x','r']);
    legend(["reconstructed binary spatial field",'sensor locations'],'Location','southeast','FontSize',15)
    title("The Reconstructed Binary Spatial Field",'FontSize', 20)
    savefig("Experiment/SyntheticExperiment/Figs/ReconstructedField.fig")
end
end