close all;clc;clear all;

pd = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);

M=50;
sn=1;
printOpt=true;
figOpt=false;
alpha=0.1;
n=2;

Psignal=pd.mean.^2+pd.var;

%% Analyze effects of signal variance on F1 score
F1_SBLUE = zeros([n,1]);
snr = 5:5:5*n;
sn_lst = sqrt(Psignal./snr); 
for i=1:n
    fprintf("Iteration %d\n",i)
    F1_SBLUE(i)=FuncSyntheticExperiment(M,sn_lst(i),alpha,printOpt,figOpt); 
end

plot(snr,F1_SBLUE)