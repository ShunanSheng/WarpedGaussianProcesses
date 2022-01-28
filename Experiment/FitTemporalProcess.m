function [hyp0, hyp1] = FitTemporalProcess(figOpt)
% given dataset, fit the temporal processes
% import data 
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

%% selecting the fields, no missing data
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
[pd_light, hyp_light, ~] = fitTemporal(Temperature_light, distname);
[pd_heavy, hyp_heavy, ~] = fitTemporal(Temperature_heavy, distname);

%% construct the hypotheses
hyp0 = struct('mean', 0 ,'cov',hyp_light.cov,'dist',pd_light);
hyp1 = struct('mean', 0 ,'cov',hyp_heavy.cov,'dist',pd_heavy);

%% plot the warping distribution and the histogram
if figOpt==true
    figure()
    qqplot(Temperature_light)
    title("Q-Q plot of Temperature when the average weekly RH is below median")  
    
    figure()
    qqplot(Temperature_light)
    title("Q-Q plot of Temperature when the average weekly RH is above median")  

    x = linspace(0, 20, 100);
    p_heavy = pdf(pd_heavy,x);
    figure();
    histogram(Temperature_heavy,'Normalization','pdf')
    hold on
    plot(x,p_heavy)
    title("Histogram of Temperature when the average weekly RH is above median ")

    p_light = pdf(pd_light,x);
    figure();
    histogram(Temperature_light,'Normalization','pdf')
    hold on
    plot(x,p_light)
    title("Histogram of Temperature when the average weekly RH is below median ")
end

end