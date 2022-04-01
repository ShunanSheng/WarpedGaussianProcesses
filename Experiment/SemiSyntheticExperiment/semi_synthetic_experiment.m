%% evaluation of the semi-synthetic experiment 
clc, clear all, close all
file_name = "Experiment/SemiSyntheticExperiment/semisyn_result.mat";
% parameters for the sensor network, physical meaning is lost
T = 19 * 7; % time period for temporal processes [0, T]
M = 19 * 7; % number of point observations, take observation at the end of 
K = 19 * 7; % number of integral observations
snP = 0.1; % signal noise of point sensors
snI = 0.1; % signal noise of integral sensors
ratio = 1; % percentage of point sensors over all sensors
alpha = 0.1;
modelHyp = struct("T", T, "M", M, "K", K, "snI", snI, "snP", snP, 'ratio', ratio, 'alpha', alpha);
Options = struct("figOpt", false, "printOpt", false, "VaryParameter", 1);
N = 100;
root = "Experiment/SemiSyntheticExperiment/semisyn_offline_";
%% compute the average F1 score and MSE
MSE = cell(N, 1);
F1 = cell(N, 1);
TPR = cell(N, 1);
FPR = cell(N, 1);

for i = 1 : N
    [MSE{i}, F1{i}, TPR{i}, FPR{i}]  = FuncSemiSyntheticExperiment(modelHyp, Options);
    fprintf("Iteration %d\n", i)
end

%%
aveMSE = aveCell(MSE)
aveF1 = aveCell(F1)
aveTPR = aveCell(TPR)
aveFPR = aveCell(FPR)

%% store the values
save(file_name, 'aveMSE', 'aveF1', 'aveTPR', 'aveFPR','-append')

%%  MSE, F1score, FPR, TPR vs snP
snP2_lst = [25:-4:1, 0.25];
snP_lst = sqrt(snP2_lst);
N = 100;
M = 19;
L = length(snP_lst);
MSE = cell(L, 1);
F1 = cell(L, 1);
TPR = cell(L, 1);
FPR = cell(L, 1);
Options.VaryParameter = 2; % change snP
for i = 1 : L
    modelHyp = struct("T", T, "M", M, "K", K, "snI", snI, "snP", snP_lst(i), ...
        'ratio', ratio, 'alpha', alpha);
    [MSE{i}, F1{i}, TPR{i}, FPR{i}] = averageScore(modelHyp, Options, N);
    fprintf("Iteration %d\n", i)
end

lstMSEsn = expandCell(MSE);
lstF1sn = expandCell(F1);
lstTPRsn = expandCell(TPR);
lstFPRsn = expandCell(FPR);

%% store the values
save(file_name, 'snP_lst','lstMSEsn', 'lstF1sn','lstTPRsn', 'lstFPRsn','-append')
%% plot
close all
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.115 .09],[.105 .03])
plot(snP2_lst,lstMSEsn.SBLUE,'-o','MarkerSize',10, 'LineWidth',1.5)
hold on
plot(snP2_lst,lstF1sn.SBLUE,'-d','MarkerSize',10,'LineWidth',1.5)
hold on
plot(snP2_lst,lstFPRsn.SBLUE,'-v','MarkerSize',10,'LineWidth',1.5)
hold on 
plot(snP2_lst,lstTPRsn.SBLUE,'-^','MarkerSize',10,'LineWidth',1.5)
set ( gca, 'xdir', 'reverse' )
grid on 
legend(["MSE","F1 score","FPR","TPR"], 'FontSize', 15, 'Location', 'northeast')
xlabel("Noise variance",'FontSize',15)
ylabel("Scores",'FontSize',15)
title("MSE, F1 score, FPR, TPR against noise variance",'FontSize',16)


%% AUC of WGPLRT versus number of integral observations
if exist('temporal_hyper.mat','file')
    hyp0 = load('temporal_hyper.mat').hyp0;
    hyp1 = load('temporal_hyper.mat').hyp1;
else
    fprintf("No files found")
    [hyp0, hyp1] = FitTemporalProcess(figOpt);
end

% temporal processes 
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

H0 = struct("meanfunc",meanfunc0,"covfunc",{covfunc0},"hyp",hyp0);
H1 = struct("meanfunc",meanfunc1,"covfunc",{covfunc1},"hyp",hyp1);

% define warping function
warpfunc = @(pd,p) invCdf(pd,p); % the inverseCDF warping function
warpinv = @(pd,p) invCdfWarp(pd,p); % inverse function of inverseCDF warping

%% Run WGPLRT 
figOpt = Options.figOpt;
printOpt = Options.printOpt; 
M_lst = sort([19:19:114, 5:1:14, 15:2:114]);
L = length(M_lst);
FPR = cell(L,1);
TPR = cell(L,1);
AUC_lst = cell(L, 1);
%%
for i = 1 : L
    f_name = strcat(root, 'M_',num2str(M_lst(i)),".mat");
    if exist(f_name,'file')
        LRT = load(f_name).LRT;
        [TPR{i}, FPR{i}] = FuncWGPLRTroc(H0, H1, T, M_lst(i), snP, printOpt,figOpt, LRT);
    else
        [TPR{i}, FPR{i}] = FuncWGPLRTroc(H0, H1, T, M_lst(i), snP, printOpt,figOpt);
    end
    AUC_lst{i}.WGPLRT = AUC(TPR{i}, FPR{i});
    fprintf("Iteration %d\n",i)
end
AUC_lst_WGPLRT = expandCell(AUC_lst).WGPLRT;

%% store the values
file_name = "Experiment/SemiSyntheticExperiment/semisyn_result.mat"
save(file_name, 'M_lst', 'AUC_lst_WGPLRT', 'TPR', 'FPR', '-append')
%% plot
close all
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.117 .09],[.105 .03])
plot(M_lst, AUC_lst_WGPLRT, '-','LineWidth',1.5)
legend({'$\sigma_{\mathrm{I}}$=0.1'},'Interpreter','latex','FontSize', 15, 'Location', 'southeast')
grid on
xlabel("Number of Point Observations, M",'FontSize',15)
ylabel("Area Under Curve (AUC)",'FontSize',15)
title("AUC against Number of Point Observations",'FontSize',17.5)

figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.115 .09],[.105 .03])
plot(FPR{L}, TPR{L}, '-','LineWidth',1.5)
h=refline(1,0);
h.LineStyle='--';
h.Color= 'r';
h.LineWidth = 1.5;
hold off
ylim([0,1.05])
legend({'WGPLRT','y=x'},'FontSize', 15, 'Location', 'southeast')
xlabel("False Positive Rate",'FontSize',15)
ylabel("True Positive Rate",'FontSize',15)
title("ROC curves when M=133",'FontSize',20)


%% SBLUE along the ROC of WGPLRT
if exist('spatial_hyper.mat','file')
    hyp_sp = load('spatial_hyper.mat').hyp_final;
    stns_loc = load('spatial_hyper.mat').stns_loc;
else
    [hyp_sp, stns_loc] = FitSpatialField(figOpt);
end

meanfunc = @meanConst; 
covfunc = {@covMaterniso, 5};
c = hyp_sp.thres;

hypSp = struct("meanfunc", meanfunc, "covfunc", {covfunc}, "hyp", hyp_sp);
warpfunc_sp = @(c,x) indicator(c,x); % the warping function of the binary spatial field is the indicator function

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

% partition the tranining and test index
% assume all sensors are point sensors, i.e., integral sensors = []
% the tranining data are the observations from 21 sensors
indexTrain = (nx * ny + 1 : size(X, 1))';
% indexTrain = [1:50:2500, (nx * ny + 1 : size(X, 1))]';
indexTest = setdiff(1:size(X,1), indexTrain)';

% partition the training and test data
Xtrain=X(indexTrain,:); 
Xtest=X(indexTest,:);

% generate the spatial field
% generate the binary spatial random field (with knowledge of the latent GP for GPR)
g = SimGP(hyp_sp,meanfunc,covfunc,X);
Y = warpfunc_sp(c,g);
Ytrain=Y(indexTrain);
Ytest=Y(indexTest);

% initialize an empty prediction array
Yhat=zeros(size(X,1),1); 

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

% construct transition matrices for all sensors
liP = ismember(indexTrain,xP);  % the locations of the point observations (using WGPLRT)
liI = ismember(indexTrain,xI);  % the locations of the integral observations (using NLRT)

%%
% select the valid range of FPR, TPR
M = 133;
f_name = strcat(root, 'M_',num2str(M),".mat");
if exist(f_name,'file')
    LRT = load(f_name).LRT;
    SBLUEprep = load(f_name).SBLUEprep;
    [TP_lst, FP_lst] = FuncWGPLRTroc(H0, H1, T, M, snP, Options.printOpt,Options.figOpt, LRT);
else
    error("The file does not exist")
end

%%
st = find(FP_lst > 0, 1,'first')
ed = find(FP_lst == 1, 1,'first')
TP = TP_lst(st : ed - 1);
FP = FP_lst(st : ed - 1);

L = length(FP);
MSE = zeros(L,1);
FPR = zeros(L,1);
TPR = zeros(L,1);

S = 1000;

n= size(X,1);
C = feval(covfunc{:}, hyp_sp.cov, X);
mu = meanfunc( hyp_sp.mean, X);
cholC = chol(C+1e-9*eye(n))';

for i=1:L % We expect when rho is extreme, i.e. close to 0 or 1
    % the performance is better since it gives more information to the
    % SBLUE
    tp = TP(i);
    fp = FP(i);
    A =[1-fp,fp;1-tp,tp]; % Define the confusion matrix for both A1,A2
    
    Yhat_noise = zeros(length(indexTrain), S);
    YT = zeros(length(indexTest), S);
    for j=1:S
        % Simulate the noisy data
        g = cholC * randn(n, 1) + mu ;
        Y = warpfunc_sp(c,g);
        Ytrain = Y(indexTrain);
        Ytest = Y(indexTest);
        
        YT(:, j) = Ytest;
        Y1 = Ytrain == 1;
        Y0 = Ytrain == 0;
        
        rnd1 = rand(length(Ytrain),1) > tp; % filp the 1 to 0 with prob 1-tp
        rnd2 = rand(length(Ytrain),1) > (1-fp);% filp the 0 to 1 with prob fp
 
        idP1 = logical(liP.*Y1);
        idP0 = logical(liP.*Y0);
        
        idI1 = logical(liI.*Y1);
        idI0 = logical(liI.*Y0);
        
        Yhat_noise(idP1,j) = (1-rnd1(idP1)).*1 + rnd1(idP1).* 0;
        Yhat_noise(idP0,j) = (1-rnd2(idP0)).*0 + rnd2(idP0).* 1;
                                              
        Yhat_noise(idI1,j) = (1-rnd1(idI1)).*1 + rnd1(idI1).* 0;
        Yhat_noise(idI0,j) = (1-rnd2(idI0)).*0 + rnd2(idI0).*1;
                                    
    end
    transitionMat = SBLUE_confusion(A,A,liP,liI);
    % Apply SBLUE 
    SBLUE = SBLUE_stats(SBLUEprep,transitionMat,c);
    Ypred = SBLUE_pred(SBLUE,Yhat_noise);
    % Evaluate the MSE and Accuracy
    MSE(i) = sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));
    [TPR(i),FPR(i)] = confusionMat(YT(:),Ypred(:));
    if mod(i,floor(L/10))==0
        fprintf("Iteration %d, tp=%4.2f, fp=%4.2f, MSE=%4.2f\n",i,tp, fp, MSE(i));
    end
end

%% plot
close 
% fig = tight_subplot(1,1,[.01 .03],[.09 .07],[.08 .03])
% plot(FP, MSE, 'LineWidth', 1.5)
% grid on 
% title("MSE of SBLUE against significance level",'FontSize',22)
% xlabel("Significance level",'FontSize',15)
% ylabel("Mean-square-error",'FontSize',15)


fig = tight_subplot(1,1,[.01 .03],[.09 .07],[.08 .03])
scatter(TP - FP, MSE, 'LineWidth', 1.5)
grid on 
title("MSE of SBLUE agaisnt TPR - FPR",'FontSize',22)
xlabel("TPR - FPR",'FontSize',15)
ylabel("Mean-square-error",'FontSize',15)


%%
SBLUE.TP = TP;
SBLUE.FP = FP;
SBLUE.MSE = MSE;
save(file_name, "SBLUE", '-append')

%% function utilities
function C_final = aveCell(C)
    f = fieldnames(C{1});
    for i = 1: numel(f)
         C_final.(f{i}) =  mean(cellfun(@(v) v.(f{i}), C));
    end
end

function C_final = expandCell(C)
    f = fieldnames(C{1});
    for i = 1: numel(f)
         C_final.(f{i}) =  cellfun(@(v) v.(f{i}), C);
    end
end

function [aveMSE, aveF1, aveTPR, aveFPR] = averageScore(modelHyp, Options, N)
    MSE = cell(N, 1);
    F1 = cell(N, 1);
    TPR = cell(N, 1);
    FPR = cell(N, 1);
    
    for i = 1 : N
        [MSE{i}, F1{i}, TPR{i}, FPR{i}]  = FuncSemiSyntheticExperiment(modelHyp, Options);
    end
    aveMSE = aveCell(MSE);
    aveF1 = aveCell(F1);
    aveTPR = aveCell(TPR);
    aveFPR = aveCell(FPR);
end

