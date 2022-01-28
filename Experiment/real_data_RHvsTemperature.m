% import data 
clc,clear all,close all
T6 = readtable('Data/Weather_Data_NEA/AWS S06 Hrly 2005-2019.csv','VariableNamingRule','modify');

% data from 5 to 23 every day in 2012, 
Data_raw = T6(T6.Year==2012,:);
Data_raw = Data_raw(Data_raw.Hour>=5 & Data_raw.Hour<=23,:);

% take days from months June to Sept (Southwest Monsoon Season)
Data_select = Data_raw(Data_raw.Month>=6 & Data_raw.Month<=9,:);

nweeks = round(size(Data_select)/19/7);
weeks = 1:nweeks;
ndays = 7 * nweeks;
nhours = 19 * ndays;
Data = Data_raw(19*2+1:19*2+nhours, :);
head(Data)

%% selecting the fields, no missing data
Rainfall_raw = Data.TotalRainfall_mm_;
RH_raw = Data.RH___;
Temperature_raw = Data.Temperature__C_;

%% compute the total rainfall over weeks
RH_reshaped = reshape(RH_raw, 19*7, []);
RH_total_week = sum(RH_reshaped, 1)./19./7;
RH_total_week = rmoutliers(RH_total_week, 'mean');

%% thresholding based on median value
RH_med = median(RH_total_week);
% RH_med = 75.6451;
RH_light_week = weeks(RH_total_week < RH_med);
RH_heavy_week = weeks(RH_total_week >= RH_med);

%% select daily maximum temperature 
% make the data right skewed (10 - data)
Temperature_reshaped = reshape(Temperature_raw, 19, []);
Temperature_hourly_ave = sum(Temperature_reshaped, 2)./size(Temperature_reshaped,2);
Temperature_ave_rm = reshape(Temperature_reshaped - Temperature_hourly_ave, [], 1);
Temperature_new = processTemp(Temperature_ave_rm);


%% partition the temperature data according to light/heavy rainfall weeks
Temperature_light_cell = cell(length(RH_light_week), 1);
Temperature_heavy_cell = cell(length(RH_heavy_week), 1);
for i = 1 : length(RH_light_week)
    week = RH_light_week(i);
    Temperature_light_cell{i} = Temperature_new(19*7*(week-1)+1:19*7*week);
end

for i = 1 : length(RH_heavy_week)
    week = RH_heavy_week(i);
    Temperature_heavy_cell{i} = Temperature_new(19*7*(week-1)+1:19*7*week);
end

Temperature_light = cell2mat(Temperature_light_cell);
Temperature_heavy = cell2mat(Temperature_heavy_cell);

%% fit the temporal gaussian process and the warping distribution
distname = "Gamma";
[pd_light, hyp_light, nlml_light] = fitTemporal(Temperature_light, distname);
[pd_heavy, hyp_heavy, nlml_heavy] = fitTemporal(Temperature_heavy, distname);


%% plot the warping distribution and the histogram
x = linspace(0, 20, 100);
p_heavy = pdf(pd_heavy,x);
figure(1);
histogram(Temperature_heavy,'Normalization','pdf')
hold on
plot(x,p_heavy)
title("Temperature when the average weekly RH is above median ")

p_light = pdf(pd_light,x);
figure(2);
histogram(Temperature_light,'Normalization','pdf')
hold on
plot(x,p_light)
title("Temperature when the average weekly RH is below median ")

%% check whether WGPLRT is able to differentiate two hypotheses
% H0
meanfunc0 = @meanConst; 
covfunc0 = {@covMaterniso, 3}; hyp0.cov = hyp_light.cov;
pd0 = pd_light;

% H1
meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 3}; hyp1.cov = hyp_light.cov;
pd1 = pd_heavy;

% Parameters for the sensor network
T = 19*7;
M = 19*7;
snP = 0.1; % each point observation zP is of size Mx1 with noise ~ N(0,snP^2I)


% Lower/upper bound for optimization in Laplace Approximation,i.e. the range of W
warpdist0 = "Gamma";warpdist1 = "Gamma";

[lb0,ub0] = lowUpBound(warpdist0,M);
[lb1,ub1] = lowUpBound(warpdist1,M);

% Hyperparameters
hyp0 = struct('mean',0,'cov',hyp0.cov,'dist',pd0,'lb',lb0,'ub',ub0);
hyp1 = struct('mean',0,'cov',hyp1.cov,'dist',pd1,'lb',lb1,'ub',ub1);

H0 = struct("meanfunc",meanfunc0,"covfunc",{covfunc0},"hyp",hyp0);
H1 = struct("meanfunc",meanfunc1,"covfunc",{covfunc1},"hyp",hyp1);

printOpt = false;
figOpt = false;
alpha = 0.1;

% run WGPLRT
[tp,fp,optLogGamma] = FuncWGPLRT(H0, H1, T, M, snP,alpha, printOpt,figOpt);




