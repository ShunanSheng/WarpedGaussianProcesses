
clc,clear all,close all
% import data 
T6 = readtable('Data/Weather_Data_NEA/S06 Hrly 2005-2019.csv','VariableNamingRule','modify');

% data from 5 to 23 every day 2005 -2017.8.20, 
Data_raw = rmmissing(T6(1:110280,:));

% take days from months April to June (Rain Season)
Data_select = Data_raw(Data_raw.Month>=4 & Data_raw.Month<=6,:);

nweeks = round(size(Data_select)/19/7);
weeks = 1:nweeks;
ndays = 7 * nweeks;
nhours = 19 * ndays;
Data = Data_raw(1:nhours, :);
head(Data)

%% selecting the fields
Rainfall_raw = Data.TotalRainfall_mm_;
RH_raw = Data.RH___;
Temperature_raw = Data.Temperature__C_;

%% compute the total rainfall over weeks
RH_reshaped = reshape(RH_raw, 19*7, []);
RH_total_week = sum(RH_reshaped, 1)./19./7;
RH_total_week = rmoutliers(RH_total_week, 'mean');

%% thresholding based on median value
RH_med = median(RH_total_week);
RH_light_week = weeks(RH_total_week < RH_med);
RH_heavy_week = weeks(RH_total_week >= RH_med);

%% partition the temperature data according to light/heavy rainfall weeks
Temperature_light_cell = cell(length(RH_light_week), 1);
Temperature_heavy_cell = cell(length(RH_heavy_week), 1);
for i = 1 : length(RH_light_week)
    week = RH_light_week(i);
    Temperature_light_cell{i} = Temperature_raw(19 * 7 * (week-1)+1 : 19 * 7 * week);
end

for i = 1 : length(RH_heavy_week)
    week = RH_heavy_week(i);
    Temperature_heavy_cell{i} = Temperature_raw(19 * 7 * (week-1)+1 : 19 * 7 * week);
end

Temperature_light = cell2mat(Temperature_light_cell);
Temperature_heavy = cell2mat(Temperature_heavy_cell);

%% select daily maximum temperature 
Temperature_light_reshaped = reshape(Temperature_light, 19, []);
Temperature_light_max = max(Temperature_light_reshaped, [] ,1)';
Temperature_heavy_reshaped = reshape(Temperature_heavy, 19, []);
Temperature_heavy_max = max(Temperature_heavy_reshaped, [] ,1)';

%% remove the linear trend and make the data right skewed (40 - data)
Temperature_light_new = processTemp(Temperature_light_max);
Temperature_heavy_new = processTemp(Temperature_heavy_max);

%% fit the temporal gaussian process and the warping distribution
distname = "Gamma";
[pd_light, hyp_light, nlml_light] = fitTemporal(Temperature_light_new, distname)
[pd_heavy, hyp_heavy, nlml_heavy] = fitTemporal(Temperature_heavy_new, distname)

%% plot the density
x = linspace(0, 20, 100);
p_heavy = pdf(pd_heavy,x);
figure(1);
histogram(Temperature_heavy_new,'Normalization','pdf')
hold on
plot(x,p_heavy)
title("Temperature when the total rainfall is above ")

