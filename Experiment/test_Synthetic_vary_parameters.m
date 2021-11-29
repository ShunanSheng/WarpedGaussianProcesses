close all;clc;clear all;

pd = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);
M = 50; % the number of observations for each sensor, assign K = M in this case
sn = 1;
printOpt = true;
figOpt = false;
alpha = 0.1;
ratio = 0.5;

%% Analyze the effects of SNR on the NLRT ROC curves by evaluating the AUC

M=66;
Psignal = pd.mean.^2 + pd.var;
snr = 5:3:5 *10;
sn_lst = sqrt(Psignal ./ snr); 
n = length(snr);
AUC_lst = zeros(n, 1);

for i = 1:n
   [TPR, FPR] = FuncNLRTroc(M, sn_lst(i), printOpt, figOpt);
    AUC_lst(i) = AUC(TPR, FPR);
end
%% Analyze the effects of SNR on the WGPLRT ROC curves by evaluating the AUC
M=66;
Psignal = pd.mean.^2 + pd.var;
snr = 5:3:5 *10;
sn_lst = sqrt(Psignal ./ snr); 
n = length(snr);
AUC_lst = zeros(n, 1);

for i = 1:n
   [TPR, FPR] = FuncNLRTroc(M, sn_lst(i), printOpt, figOpt);
    AUC_lst(i) = AUC(TPR, FPR);
end

%%
plot(snr, AUC_lst,'-^')

%% Analyze the effects of the number of integral observations in NLRT on the
% ROC curves by evaluating the AUC
sn = 0.01;
M_lst = [10: 2: 64, 66: 1: 76, 78: 3 :129];
n = length(M_lst)
AUC_lst = zeros(n, 1);

for i = 1:n
    fprintf("Iteration %d \n ", i);
    [TPR, FPR] = FuncNLRTroc(M_lst(i), sn, printOpt, figOpt);
    AUC_lst(i) = AUC(TPR, FPR);
end

plot(M_lst, AUC_lst)

%% Analyze the effects of the number of integral observations in NLRT on the ROC curves
% Optimal number of observations is K = 70
n = 6;
sn = 0.1;
M_lst = 10:20:10+20*(n-1);
cell_TPR = cell(1, n);
cell_FPR = cell(1, n);
FigLegend = cell(1, n);

for i = 1:n
    fprintf("Iteration %d \n ", i);
    [cell_TPR{i}, cell_FPR{i}] = FuncNLRTroc(M_lst(i), sn, printOpt, figOpt);
    FigLegend{i}="K =" + M_lst(i);
end
% convert cell array to matrix for plotting
TP = cell2mat(cell_TPR);
FP = cell2mat(cell_FPR);

% plot the ROC graph
plotROC(TP,FP,"ROC curves against the number of integral observations K",FigLegend)

%% Analyze the effects of the number of point observations in WGPLRT on the ROC curves
n = 5;
sn = 0.1;
M_lst = 10:20:10+20*(n-1);
cell_TPR = cell(1, n);
cell_FPR = cell(1, n);
FigLegend = cell(1, n);

for i = 1:n
    fprintf("Iteration %d \n ", i);
    [cell_TPR{i}, cell_FPR{i}] = FuncWGPLRTroc(M_lst(i), sn, printOpt, figOpt);
    FigLegend{i}="M =" + M_lst(i);
end
% convert cell array to matrix for plotting
TP = cell2mat(cell_TPR);
FP = cell2mat(cell_FPR);

% plot the ROC graph
plotROC(TP,FP,"ROC curves against the number of point observations M",FigLegend)

%% Compare the ROC curve when K = M = 66 for WGPLRT and NRLT
sn = 0.1;
cell_TPR = cell(1, 2);
cell_FPR = cell(1, 2);

[cell_TPR{1}, cell_FPR{1}] = FuncWGPLRTroc(66, sn, printOpt, figOpt);
[cell_TPR{2}, cell_FPR{2}] = FuncNLRTroc(66, sn, printOpt, figOpt);
FigLegend = {"WGPLRT", "NLRT"};

% convert cell array to matrix for plotting
TP = cell2mat(cell_TPR);
FP = cell2mat(cell_FPR);

% plot the ROC graph
plotROC(TP,FP,"Receiver operating characteristic curves when M=K=70",FigLegend)


%% Analyze effects of signal variance on F1 score, TPR, and FPR
n = 10;
Psignal = pd.mean.^2 + pd.var;
F1_SBLUE = zeros([n, 1]);
TPR = zeros([n, 1]);
FPR = zeros([n, 1]);
snr = 5:5:5 *n;
sn_lst = sqrt(Psignal ./ snr); 
for i = 1:n
    fprintf("Iteration %d\n",i)
    [F1_SBLUE(i), TPR(i), FPR(i)] = FuncSyntheticExperiment(M,sn_lst(i),alpha,ratio,printOpt,figOpt); 
end

%% Plot the figure
figure()
plot(snr,F1_SBLUE,'-x')
hold on
plot(snr,FPR,'-^')
hold on
plot(snr,TPR,'-v')
legend("F1 score","FPR","TPR")
xlabel("Signal-to-noise ratio (dB)")
ylabel("Score")
title("Scores against varying SNR")
%% Analyze effects of the number of observations on F1 score, TPR, and FPR
clc
F1_SBLUE = zeros([n,1]);
TPR = zeros([n,1]);
FPR = zeros([n,1]);
sn = 0.1;
M_lst=10:10:10*n;
for i=1:n
    fprintf("Iteration %d\n",i);
    [F1_SBLUE(i),TPR(i),FPR(i)] = FuncSyntheticExperiment(M_lst(i),sn,alpha,ratio,printOpt,figOpt); 
end
%% Plot the figure
figure()
plot(M_lst,F1_SBLUE,'-x')
hold on
plot(M_lst,FPR,'-^')
hold on
plot(M_lst,TPR,'-v')
legend("F1 score","FPR","TPR")
xlabel("M, the number of observations in LRTs")
ylabel("Score")
title("Scores against varying number of observations in LRTs")

%% Analyze the effects of varying ratio of point and integral sensors
sn = 0.1;
M = 50;
ratio_lst = [0,0.25,0.5,0.75,1];
n=length(ratio_lst);
F1_SBLUE = zeros([n,1]);
TPR = zeros([n,1]);
FPR = zeros([n,1]);
for i=1:n
    fprintf("Iteration %d\n",i);
    [F1_SBLUE(i),TPR(i),FPR(i)] = FuncSyntheticExperiment(M,sn,alpha,ratio_lst(i),printOpt,figOpt); 
end

%% Plot the figure
figure()
plot(ratio_lst,F1_SBLUE,'-x')
hold on
plot(ratio_lst,FPR,'-^')
hold on
plot(ratio_lst,TPR,'-v')
legend("F1 score","FPR","TPR")
xlabel("The percentage point sensors over all senssors")
ylabel("Score")
title("Scores against varying percentage of point sensors")






