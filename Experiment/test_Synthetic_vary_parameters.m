close all;clc;clear all;

pd = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);
M=50;
sn=1;
printOpt=true;
figOpt=false;
alpha=0.1;
ratio=0.5;
n=10;

%% Analyze effects of signal variance on F1 score, TPR, and FPR
Psignal=pd.mean.^2+pd.var;
F1_SBLUE = zeros([n,1]);
TPR = zeros([n,1]);
FPR = zeros([n,1]);
snr = 5:5:5*n;
sn_lst = sqrt(Psignal./snr); 
for i=1:n
    fprintf("Iteration %d\n",i)
    [F1_SBLUE(i),TPR(i),FPR(i)] = FuncSyntheticExperiment(M,sn_lst(i),alpha,ratio,printOpt,figOpt); 
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
%% Analyze effects of number of observations on F1 score, TPR, and FPR
clc
F1_SBLUE = zeros([n,1]);
TPR = zeros([n,1]);
FPR = zeros([n,1]);
sn = 0.1;
M_lst=5:1:5+1*n;
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






