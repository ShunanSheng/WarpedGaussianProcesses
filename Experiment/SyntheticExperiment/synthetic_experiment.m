%%% Synthetic Experiments
close all;clc;clear all;
file_name = "Experiment/SyntheticExperiment/Results/syn_result.mat";
%% The parameters
T = 20; 
M = 50; 
K = 50; 
snP = 0.1; 
snI = 0.1;
ratio = 0.5;
alpha = 0.1;
modelHyp = struct("T", T,"M",M,"K",K,"snI",snI,"snP",snP,'ratio',ratio, 'alpha', alpha);
Options = struct("figOpt", false, "printOpt", false, "VaryParameter", 1, "Time", 1);
pd = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

%% Experiment 1 - Compute the average F1 score and MSE
N = 100;
MSE = cell(N, 1);
F1 = cell(N, 1);
FPR = cell(N,1);
TPR = cell(N,1);
time = cell(N,1);
for i = 1 : N
    [MSE{i}, F1{i}, TPR{i}, FPR{i}]  = FuncSyntheticExperiment(modelHyp, Options);
    fprintf("Iteration %d \n ", i);
end
aveMSE = aveCell(MSE);
aveF1 = aveCell(F1);
aveTPR = aveCell(TPR);
aveFPR = aveCell(FPR);

%% save results
save(file_name, 'aveMSE', 'aveF1','aveTPR','aveFPR','-append')


%% Experiment 1.5 - Compute the average online computational time 
N = 100;
Options.Time = 1;
OnlineTime = cell(N,1);
OfflineTime = cell(N,1);
for i = 1 : N
    [OnlineTime{i}, OfflineTime{i}]  = TimeFuncSyntheticExperiment(modelHyp, Options);
    fprintf("Iteration %d \n ", i);
end
aveOnlineTime = aveCell(OnlineTime);
aveOfflineTime = aveCell(OfflineTime);
%% Offline computational time
% Options.Time = 1;
% [~, OfflineTime] = TimeFuncSyntheticExperiment(modelHyp, Options);

%% save results
save(file_name,'aveOnlineTime', 'OfflineTime', '-append')

%% Experiment 2 - Analyze the effects of the noise variance on F1 score, MSE, TPR, and FPR
N = 10;
snP2_lst = [2.5:-0.4:0.1, 0.01, 0.0001];
snP_lst = sqrt(snP2_lst);
L = length(snP_lst);
MSE = cell(L, 1);
F1 = cell(L, 1);
FPR = cell(L,1);
TPR = cell(L,1);

Options.VaryParameter = 2;

for i = 1:L
    fprintf("Iteration %d\n",i)
    modelHyp.snP = snP_lst(i);
    modelHyp.snI = snP_lst(i);
    [MSE{i}, F1{i}, TPR{i}, FPR{i}] = averageScore(modelHyp, Options, N);
end

lstMSE = expandCell(MSE);
lstF1 = expandCell(F1);
lstTPR = expandCell(TPR);
lstFPR = expandCell(FPR);

%% save results
save(file_name,'snP_lst','lstMSE','lstF1','lstTPR','lstFPR','-append')

%% plot
close
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.115 .07],[.095 .015])
plot(snP2_lst,lstMSE.SBLUE,'-o','MarkerSize',10, 'LineWidth',1.5)
hold on
plot(snP2_lst,lstF1.SBLUE,'-d','MarkerSize',10,'LineWidth',1.5)
hold on
plot(snP2_lst,lstFPR.SBLUE,'-v','MarkerSize',10,'LineWidth',1.5)
hold on 
plot(snP2_lst,lstTPR.SBLUE,'-^','MarkerSize',10,'LineWidth',1.5)
hold off
grid on 
set (gca, 'xdir', 'reverse' )
legend(["MSE","F1 score","FPR","TPR"], 'FontSize', 15, 'Location', 'northwest')
xlabel("Noise variance",'FontSize',15)
ylabel("Scores",'FontSize',15)
title("MSE, F1 score, FPR, TPR against noise variance",'FontSize',16)
savefig("Experiment/SyntheticExperiment/Figs/ScoresVsSn.fig")
%% Initialize for analysis of WGPLRT, NLRT
% Parameters for the sensor network
T = modelHyp.T; % time period for temporal processes [0, T]
M = modelHyp.M; % number of point observations, take observation at the end of 
K = modelHyp.K; % number of integral observations
snP = modelHyp.snP; % signal noise of point sensors
snI = modelHyp.snI; % signal noise of integral sensors
ratio = modelHyp.ratio; % percentage of point sensors over all sensors
alpha = modelHyp.alpha; % the significance level for LRT

% H0 Null hypothesis
meanfunc0 = @meanConst; 
covfunc0 = {@covMaterniso, 1}; ell0 = 1; sf0 = 1; hyp0.cov = log([ell0; sf0]);
pd0 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

% H1 Alternative hypothesis
meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 5}; ell1 = 1; sf1 = 1; hyp1.cov = log([ell1; sf1]);
pd1 = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

% Lower/upper bound for optimization in Laplace Approximation,i.e. the
% range of W, here is R, same as the range for when the distribution is
% normal
warpdist0 = "Normal"; 
warpdist1 = "Normal";

[lb0,ub0] = lowUpBound(warpdist0,M);
[lb1,ub1] = lowUpBound(warpdist1,M);

% Create structures to store the hyperparameters
hyp0 = struct('mean',0,'cov',hyp0.cov,'dist',pd0,'t',T,'lb',lb0,'ub',ub0);
hyp1 = struct('mean',0,'cov',hyp1.cov,'dist',pd1,'t',T,'lb',lb1,'ub',ub1);

H0 = struct("meanfunc",meanfunc0,"covfunc",{covfunc0},"hyp",hyp0);
H1 = struct("meanfunc",meanfunc1,"covfunc",{covfunc1},"hyp",hyp1);

% warping function
warpfunc = @(pd,p) invCdf(pd,p);
warpinv = @(pd,p) invCdfWarp(pd,p);
figOpt = Options.figOpt;
printOpt = Options.printOpt; 

%% Experiment 3 - Analyze the effects of noise variance on the WGPLRT ROC curves by evaluating the AUC
M = 50;
snP2_lst = [2.5:-0.4:0.1, 0.01, 0.0001];
snP_lst = sqrt(snP2_lst);
% snP_lst = 1./(1:5:50);
L = length(snP_lst) ;
FPR = cell(L,1);
TPR = cell(L,1);
AUC_lst = cell(L, 1);

for i = 1 : L
   [TPR{i}, FPR{i}] = FuncWGPLRTroc(H0, H1, T, M, snP_lst(i), printOpt,figOpt);
    AUC_lst{i}.WGPLRT = AUC(TPR{i}, FPR{i});
    fprintf("Iteration %d\n",i)
end

%% Experiment 4 - Analyze the effects of noise variance on the NLRT ROC curves by evaluating the AUC
K = 50;
snI2_lst = [2.5:-0.4:0.1, 0.01, 0.0001];
snI_lst = sqrt(snI2_lst);
L = length(snI_lst);
FPR = cell(L,1);
TPR = cell(L,1);

for i = 1 : L
   [TPR{i}, FPR{i}] = FuncNLRTroc(H0, H1, T, K, snI_lst(i), printOpt,figOpt);
    AUC_lst{i}.NLRT = AUC(TPR{i}, FPR{i});
    fprintf("Iteration %d\n",i)
end
AUC_sn = expandCell(AUC_lst);
%% save results
save(file_name, 'AUC_sn', '-append')
%% plot
close
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.12 .08],[.11 .015])
plot(snI2_lst, AUC_sn.WGPLRT,'-^','MarkerSize',10,'LineWidth',1.5)
hold on
plot(snI2_lst, AUC_sn.NLRT, '-v','MarkerSize',10,'LineWidth',1.5)
hold off
grid on 
set (gca, 'xdir', 'reverse' )
ylim('auto')
legend({'WGPLRT','NLRT'},'FontSize', 15, 'Location', 'southeast')
xlabel("Noise variance",'FontSize',15)
ylabel("Area Under Curve (AUC)",'FontSize',15)
title("AUC against noise variance",'FontSize',20)
% savefig("Experiment/SyntheticExperiment/Figs/AUC_Sn.fig")

%%
close
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.12 .08],[.11 .015])
plot(1:5:50, AUC_sn.WGPLRT,'-^','MarkerSize',10,'LineWidth',1.5)
hold on
plot(1:5:50, AUC_sn.NLRT, '-v','MarkerSize',10,'LineWidth',1.5)
hold off
grid on 
ylim('auto')
legend({'WGPLRT','NLRT'},'FontSize', 15, 'Location', 'southeast')
xlabel("Noise variance",'FontSize',15)
ylabel("Area Under Curve (AUC)",'FontSize',15)
title("AUC against noise variance",'FontSize',20)

%% Experiment 5 - Analyze the effects of the number of integral observations in NLRT on the 
%  ROC curves by evaluating the AUC
snI = 0.01;
K_lst = 10: 3: 129;
L = length(K_lst);
FPR = cell(L,1);
TPR = cell(L,1);
AUC_lst_K = cell(L,1);

for i = 1 : L
    fprintf("Iteration %d \n ", i);
    [TPR{i}, FPR{i}] = FuncNLRTroc(H0, H1, T, K_lst(i), snI, printOpt,figOpt);
    AUC_lst_K{i}.sn001 = AUC(TPR{i}, FPR{i});
end

snI = 0.1;
L = length(K_lst);
FPR = cell(L,1);
TPR = cell(L,1);

for i = 1 : L
    fprintf("Iteration %d \n ", i);
    [TPR{i}, FPR{i}] = FuncNLRTroc(H0, H1, T, K_lst(i), snI, printOpt,figOpt);
    AUC_lst_K{i}.sn01 = AUC(TPR{i}, FPR{i});
end
AUC_lst_NLRT_K = expandCell(AUC_lst_K);
%% Experiment 6 - Analyze the effects of the number of point observations in WGPLRT on the 
% ROC curves by evaluating the AUC
snP = 0.01;
M_lst = 10: 3: 129;
L = length(M_lst);
FPR = cell(L,1);
TPR = cell(L,1);
AUC_lst_M = cell(L,1);

for i = 1 : L
    fprintf("Iteration %d \n ", i);
    [TPR{i}, FPR{i}] = FuncWGPLRTroc(H0, H1, T, M_lst(i), snP, printOpt,figOpt);
    AUC_lst_M{i}.sn001 = AUC(TPR{i}, FPR{i});
end

snP = 0.1;
L = length(M_lst);
FPR = cell(L,1);
TPR = cell(L,1);

for i = 1 : L
    fprintf("Iteration %d \n ", i);
    [TPR{i}, FPR{i}] = FuncWGPLRTroc(H0, H1, T, M_lst(i), snP, printOpt,figOpt);
    AUC_lst_M{i}.sn01 = AUC(TPR{i}, FPR{i});
end
AUC_lst_WGPLRT_M = expandCell(AUC_lst_M);


%% save results
save(file_name, 'K_lst','M_lst', 'AUC_lst_NLRT_K','AUC_lst_WGPLRT_M','-append')

%% plot
close
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
plot(K_lst, AUC_lst_NLRT_K.sn001,'-b','LineWidth',1.5)
hold on 
plot(K_lst, AUC_lst_NLRT_K.sn01,'--r','LineWidth',1.5)
plot(M_lst, AUC_lst_WGPLRT_M.sn001,'-o','LineWidth',1.5)
plot(M_lst, AUC_lst_WGPLRT_M.sn01,'-*','LineWidth',1.5)
hold off
grid on 
ylim([0.5,1.05]) 
legend({'NLRT $\sigma_{\mathrm{I}}$=0.01','NLRT $\sigma_{\mathrm{I}}$=0.1', ...
    'WGPLRT $\sigma_{\mathrm{I}}$=0.01','WGPLRT $\sigma_{\mathrm{I}}$=0.1'},'Interpreter','latex','FontSize', 15, 'Location', 'southeast')
xlabel("Number of observations",'FontSize',15)
ylabel("Area Under Curve (AUC)",'FontSize',15)
title("AUC against Number of Observations",'FontSize',16.5)
savefig("Experiment/SyntheticExperiment/Figs/AUCvaryingK.fig")

%% Experiment 7 - Compare the ROC curve when K = M = 50 for WGPLRT and NRLT
sn = 0.1;
cell_TPR = cell(1, 2);
cell_FPR = cell(1, 2);
K = 50;
M = 50;
figOpt = true;
printOpt = false;

[cell_TPR{1}, cell_FPR{1}] = FuncNLRTroc(H0, H1, T, K, sn, printOpt,figOpt);
[cell_TPR{2}, cell_FPR{2}] = FuncWGPLRTroc(H0, H1, T, M, sn, printOpt,figOpt);

% convert cell array to matrix for plotting
TP = cell2mat(cell_TPR);
FP = cell2mat(cell_FPR);

%% save the results
save(file_name, 'cell_TPR','cell_FPR','-append')
%% plot the ROC graphs
close all
figure('Position',[100,100,400,300])
tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
plot(cell_FPR{1}, cell_TPR{1},'-','color','#0072BD', 'MarkerSize',10,'LineWidth',1.5)
hold on
plot(cell_FPR{2}, cell_TPR{2},'-','MarkerSize',10,'LineWidth',1.5)
hold on 
h=refline(1,0);
h.LineStyle='--';
h.Color= 'r';
h.LineWidth = 1.5;
hold off
ylim([0,1.05])
grid on
legend({'NLRT','WGPLRT','y=x'},'FontSize', 15, 'Location', 'southeast')
xlabel("False positive rate",'FontSize',15)
ylabel("True positive rate",'FontSize',15)
title("ROC curves when M=K=50",'FontSize',20)
savefig("Experiment/SyntheticExperiment/Figs/ROC_at_OptM=50.fig")

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
        [MSE{i}, F1{i}, TPR{i}, FPR{i}]  = FuncSyntheticExperiment(modelHyp, Options);
    end
    aveMSE = aveCell(MSE);
    aveF1 = aveCell(F1);
    aveTPR = aveCell(TPR);
    aveFPR = aveCell(FPR);
end


