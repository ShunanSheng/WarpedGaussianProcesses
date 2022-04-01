clc,clear all,close all
T6 = readtable('Data/RealDataSet/S06 Hrly 2005-2019.csv','VariableNamingRule','modify');
% T23 = readtable('Data/RealDataSet/S23 Hrly 2005-2019.csv','VariableNamingRule','modify');
% T24 = readtable('Data/RealDataSet/S24 Hrly 2005-2019.csv','VariableNamingRule','modify');

Data_raw = rmmissing(T6);
% Data_raw = rmmissing(T23);
% Data_raw = rmmissing(T24);


nweeks =104;
weeks = 1:nweeks;
ndays = 7 * nweeks;
nhours = 19 * ndays;
Data = tail(Data_raw, nhours);

head(Data)

Rainfall_raw = Data.TotalRainfall_mm_;
Temperature_raw = Data.Temperature__C_;
%% compute the total rainfall over weeks
Rainfall_reshaped = reshape(Rainfall_raw, 19*7, []);
Rainfall_total_week = sum(Rainfall_reshaped, 1);
Rainfall_total_week = rmoutliers(Rainfall_total_week, 'mean');

%% thresholding based on median value
Rainfall_med = median(Rainfall_total_week);
Rainfall_light_week = weeks(Rainfall_total_week < Rainfall_med);
Rainfall_heavy_week = weeks(Rainfall_total_week >= Rainfall_med);

%% partition the temperature data according to light/heavy rainfall weeks
Temperature_light_cell = cell(length(Rainfall_light_week), 1);
Temperature_heavy_cell = cell(length(Rainfall_heavy_week), 1);
for i = 1 : length(Rainfall_light_week)
    week = Rainfall_light_week(i);
    Temperature_light_cell{i} = Temperature_raw(19 * 7 * (week-1)+1 : 19 * 7 * week);
end

for i = 1 : length(Rainfall_heavy_week)
    week = Rainfall_heavy_week(i);
    Temperature_heavy_cell{i} = Temperature_raw(19 * 7 * (week-1)+1 : 19 * 7 * week);
end

Temperature_light = cell2mat(Temperature_light_cell);
Temperature_heavy = cell2mat(Temperature_heavy_cell);

%%
distname = "Gamma";
[pd_light, hyp_light, nlml_light] = fitTemporal(Temperature_light(1:100), distname);
[pd_heavy, hyp_heavy, nlml_heavy] = fitTemporal(Temperature_heavy(1:100), distname);

%% fit the warping functions & gaussian processes
pd_light = fitdist(RH_light,'Beta')
pd_heavy = fitdist(RH_heavy,'Beta')

%% plot the fitted density plot and histogram
x = linspace(0, 1, 100);
p_heavy = pdf(pd_heavy,x);
figure(1);
histogram(RH_heavy,'Normalization','pdf')
hold on
plot(x,p_heavy)
title("Temperature when the total rainfall is above ")

p_light = pdf(pd_light,x);
figure(2);
histogram(RH_light,'Normalization','pdf')
hold on
plot(x,p_light)
title("Temperature when the total rainfall is below ")

%% fit the gaussian process, also see data_exploration.m
gp_temp_light = norminv(cdf(pd_light, RH_light));
gp_temp_heavy = norminv(cdf(pd_heavy, RH_heavy));


%%
y = gp_temp_light(1:1000);
x = (1:length(y))';

meanfunc = @meanConst; hyp.mean = 0;
ell = 1; sf = 1;  
covfunc = {@covMaterniso, 3}
hyp.cov = log([ell;sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

prior.mean = {{@priorDelta}};
prior.lik = {{@priorDelta}};
prior.cov = {[];{@priorDelta}};
inf = {@infPrior,@infGaussLik,prior};

hyp2 = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x, y);
nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)

%%



