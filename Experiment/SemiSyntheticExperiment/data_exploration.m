clc;clear all;

% Temperature_max = max(Temperature_reshaped, [] ,1)';
% Temperature_new = processTemp(Temperature_max);
Temperature_reshaped = reshape(Temperature_raw, 19, []);
Temperature_ave = sum(Temperature_reshaped, 2)./size(Temperature_reshaped,2);

C = {'0.000000'; '10.000000'; '100000.000000'};
M = str2double(C)
%%
test_cell={double([2,1]'),double([2,1]')};

%%
t = table({1:24}, {1:48}, {1:48}, {1:48}); 
t = reshape(t, 2)



%%
T = readtable('Data/Weather_Data_NEA/S06 Hrly 2005-2019.csv','VariableNamingRule','modify');



%%


clc,close all
Data_raw = T;
% Data_raw = rmmissing(T);
head(Data_raw)
size(Data_raw)
%%
Data = Data_raw(Data_raw.Month==3,:);
%%
Temperature_raw = Data.Temperature__C_;
Temperature_raw(isnan(Temperature_raw)) = 0;
Temperature_reshaped = reshape(Temperature_raw, 24, []);
Temperature_max = max(Temperature_reshaped, [] ,1)';
% Temperature_mean = mean(Temperature_reshaped, [] ,1))

%% we should not remove outliers in this case

Temperature_flipped = 40 - Temperature_max;
pd = fitdist(Temperature_flipped, 'Gamma');
x = linspace(0, 50, 100)';
p_heavy = pdf(pd,x);
figure(1);
histogram(Temperature_flipped,'Normalization','pdf')
hold on
plot(x,p_heavy)
title("Temperature when the total rainfall is above ")



%% 
ndays =200
nhours =19 * 200;
Data = Data_raw(1:nhours,:);

%% we want to model spatial field: RH
% temporal processes: Temperarture
Rainfall_raw = Data.TotalRainfall_mm_;
Temperature_raw = Data.Temperature__C_;
Cloud_cover_raw = Data.TotalCloudCover;
RH_raw = Data.RH___;

%%  
RH_reshaped = reshape(RH_raw,19,[]);
RH_average =  mean(RH_reshaped,1)';

%% 
y = RH_average;
x = (1:length(y))';

meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];

covfunc = {@covSEiso};
likfunc = @likGauss; sn = 0.001; 

% prior.mean = {{@priorDelta}};
% prior.lik = {{@priorDelta}};


ell = 1; sf = 1; 
hyp.cov = log([ell;sf]);
prior.cov = {[];[]};

hyp.lik = log(sn);

% covfunc = {@covSum, {@covSEiso, @covPeriodic}}; 
% ell = 1; sf = sqrt(2/3);  
% ellPeriodic = 1 ; pPeriodic = 2; sfPeriodic = sqrt(1/3);
% hyp.cov = log([ell;sf;ellPeriodic;pPeriodic;sfPeriodic]);
% prior.cov = {[];{@priorDelta};[];[];{@priorDelta}};

inf = {@infPrior,@infGaussLik,prior};
nlml = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y)


%% training
hyp2 = minimize(hyp, @gp, -1000, inf, meanfunc, covfunc, likfunc, x, y);
nlml2 = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
%%
n = length(y);
K2 = feval(covfunc{:}, hyp2.cov, x);
mu2 = feval(meanfunc{:}, hyp2.mean, x);
% y2 = chol(K2 + exp(hyp2.lik).^2 * eye(n))'*randn(n, 1) + mu2;
s2 = diag(K2) + exp(hyp2.lik).^2 * ones(n,1);

%%
f = [mu2+2*sqrt(s2); flipdim(mu2-2*sqrt(s2),1)];
fill([x; flipdim(x,1)], f, [7 7 7]/8)
hold on; plot(x, mu2); plot(x, y, 'r');
% figure()
% plot(x,y,'r')
% hold on
% plot(x,y2,'b')



%% remove outliers  
% boxplot(Temperature_raw);
boxplot(RH_raw);

[Temperature, index] = rmoutliers(Temperature_raw,'mean');
RH = rmoutliers(RH_raw(~index),'mean');
% Rainfall = rmoutliers(Rainfall_raw,'mean');

%% 

meanfunc = @meanConst; hyp.mean = 0;
covfunc = {@covSEiso};
sn = 0.01; hyp.lik = log(sn);
ell = 1; sf = 1; 
hyp.cov = log([ell;sf]);

n = 50;
x = linspace(-10,10,n)';
K = feval(covfunc{:}, hyp.cov, x);
mu = feval(meanfunc, hyp.mean, x);
y = chol(K)'*randn(n, 1) + mu + exp(hyp.lik)* randn(n, 1);

plot(x,y)
%%

alpha = 0.05;
[H,p_value] = Mann_Kendall(y,alpha)



%% normality test
% qqplot(Rainfall);
qqplot(Temperature);
qqplot(RH);

%% partition based to have two different 
med = median(RH);
index_low = RH <= med;
index_high = RH > med;

low_RH = RH(index_low);
high_RH = RH(index_high);

low_temp = Temperature(index_low);
high_temp = Temperature(index_high);

% low_rain = Rainfall(index_low);
% high_rain = Rainfall(index_high);

%% fitting the probability distribution


pd = fitdist(low_temp,'Gamma');

%%
figure();
histogram(low_temp, 'Normalization', 'pdf')
hold on
x_values = (20:0.1:38);
y = pdf(pd, x_values);
plot(x_values, y)
hold off

%%

low_temp_subset = low_temp(1:2000);
pd = fitdist(low_temp_subset,'Gamma');
gp_temp_raw = norminv(cdf(pd, low_temp_subset));
%%
plot(gp_temp_raw(:))

%% 
gp_temp = gp_temp_raw(:);

%% 
clc
y = gp_temp;
x = (1:length(y))';

%%
clc
meanfunc = @meanConst; hyp.mean = 0;

covfunc = {@covSEiso};
likfunc = @likGauss; sn = 0.001; hyp.lik = log(sn);
prior.mean = {{@priorDelta}};
% prior.lik = {{@priorDelta}};


% ell = 1; sf = 1; 
% hyp.cov = log([ell;sf]);
% prior.cov = {[];{@priorDelta}};

covfunc = {@covSum, {@covSEiso, @covPeriodic}}; 
ell = 1; sf = sqrt(2/3);  
ellPeriodic = 1 ; pPeriodic = 2; sfPeriodic = sqrt(1/3);
hyp.cov = log([ell;sf;ellPeriodic;pPeriodic;sfPeriodic]);
prior.cov = {[];{@priorDelta};[];[];{@priorDelta}};

inf = {@infPrior,@infGaussLik,prior};

%%

hyp2 = minimize(hyp, @gp, -1000, inf, meanfunc, covfunc, likfunc, x, y);

%%
nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)

%%
n = length(y);
K2 = feval(covfunc{:}, hyp2.cov, x);
mu2 = feval(meanfunc, hyp2.mean, x);
y2 = chol(K2)'*randn(n, 1) + mu2 + exp(hyp2.lik)* randn(n, 1);



figure()
plot(x,y,'r')
hold on
plot(x,y2,'b')




%% fit gaussian process
clc, clear all
meanfunc = @meanConst; hyp.mean = 0;
ell = 1; sf = 1;  

covfunc = {@covSEiso};
hyp.cov = log([ell;sf]);

% covfunc = {@covSum, {@covSEiso, @covPeriodic}}; 
% ellPeriodic = 1 ; pPeriodic = 2; sfPeriodic = sqrt(2)/2;
% hyp.cov = log([ell;sf;ellPeriodic;pPeriodic;sfPeriodic]);

likfunc = @likGauss; sn = 0.0001; hyp.lik = log(sn);

n = 50;
x = linspace(-10,10,n)';
K = feval(covfunc{:}, hyp.cov, x);
mu = feval(meanfunc, hyp.mean, x);
y = chol(K)'*randn(n, 1) + mu + exp(hyp.lik)*randn(n, 1);

nlml = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y)


%%

prior.mean = {{@priorDelta}};
prior.lik = {{@priorDelta}};
prior.cov = {[];{@priorDelta}};
inf = {@infPrior,@infGaussLik,prior};

hyp2 = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x, y);

nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)

% hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
%%
K2 = feval(covfunc{:}, hyp2.cov, x);
mu2 = feval(meanfunc, hyp2.mean, x);
y2 = chol(K2)'*gpml_randn(0.15, n, 1) + mu2 + exp(hyp2.lik)*gpml_randn(0.2, n, 1);

figure()
plot(x,y,'r')
hold on
plot(x,y2,'b')

%% some random stuffs
clear all;close all;clc;

meanfunc = @meanConst;hyp.mean=0;
covfunc = {@covMaterniso, 3}; ell1=2; sf1=1; hyp.cov=log([ell1; sf1]);
pd = makedist('Gamma','a',2,'b',4);
warpfunc=@(pd,p) invCdf(pd,p);

N = 1000;
x = linspace(-100,100,N)';
hyp=struct('mean',hyp.mean,'cov',hyp.cov,'dist',pd);
g = SimGP(hyp,meanfunc,covfunc,x);
% z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);

figure()
autocorr(g,500)
figure()
plot(g)

%% test for presence of linear trend
alpha = 0.05;
[H,p_value] = Mann_Kendall(g,alpha)

b = polyfit(1:200, Temperature_raw(1:200),1)

yhat = polyval(b,[0 1:200]);
figure()
plot(Temperature_raw(1:200))
plot(x, yhat)


%% remove seasonality
mean_temp = movmean(Temperature,24);
s = 5000;

figure()
plot(Temperature(1:s))
hold on 
plot(mean_temp(1:s),'r')
hold off 


%% autocorrelation plot
autocorr(Temperature,200)

% %% stationarity
% adftest(Rainfall);
% adftest(Temperature);
% adftest(RH);
% 
% %% test for presence of linear trend
% alpha = 0.05;
% [H,p_value] = Mann_Kendall(RH,alpha)















