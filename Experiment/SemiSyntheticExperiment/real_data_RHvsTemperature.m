
clc,clear all,close all
%% import data 
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
rootdir = 'Data/Weather_Data_NEA';            % set the root directory location (could use uigetfile)
d=dir(fullfile(rootdir,'*.csv'));  % and pick up all the .csv files therein...

%% choose data from 2012.6 to 2012.9 when no entries of RH or Temperature are missing
nstns = numel(d);
RH_cell = cell(1,nstns);
Temperature_cell = cell(1,nstns);
stns_lst = cell(nstns,1);
for i = 1:nstns
    stn_name = extractBetween(d(i).name, "AWS","Hrly");
    stns_lst{i} = stn_name{1};
    data = readtable(fullfile(rootdir,d(i).name));  % read the jth file in ith directory
    data_filtered = data(data.Month >=6 &  data.Month <=9 & data.Year==2012 &...
    data.Hour>=5 & data.Hour<=23, :);
    data_RH = data_filtered.RH___;
    data_Temperature = data_filtered.Temperature__C_;
    if iscell(data_RH)
        data_RH = str2double(data_RH);
    end
    if iscell(data_Temperature)
        data_Temperature = str2double(data_Temperature);
    end
    if (sum(isnan(data_RH))==1 || sum(isnan(data_Temperature))==1)
        disp("station", stn_name)
        warning("Missing data")
    end
    RH_cell{i} = data_RH;  
    Temperature_cell{i} = data_Temperature;
end
%% create data table
nweeks = floor(size(RH_cell{1},1)/19/7);
RH = cat(2, RH_cell{:});
RH = RH(19*2+1:19*2+nweeks*19*7, :);
Temperature = cat(2, Temperature_cell{:});
Temperature = Temperature(19*2+1:19*2+nweeks*19*7, :);

%% compute average weekly RH 
weeks = 1:nweeks;
RH_reshaped = reshape(RH, 19*7, nweeks, 21);
RH_ave_week = reshape(mean(RH_reshaped, 1), nweeks, nstns);
%% get weeks with light/heavy average weekly RH 
RH_med = median(RH_ave_week, 'all');
RH_light_week_index = RH_ave_week < RH_med;
RH_heavy_week_index = RH_ave_week >= RH_med;
%% process Temperature data
% remove hourly mean and make data right-skewed
Temperature_reshaped = reshape(Temperature, 19, [], nstns);
Temperature_hourly_ave = reshape(mean(mean(Temperature_reshaped, 2),3),19,[]);
Temperature_ave_rm = reshape(Temperature_reshaped - Temperature_hourly_ave, 19 * 7 * nweeks, nstns);
Temperature_transformed = 10 - Temperature_ave_rm;

%% partition the temperature data according to light/heavy rainfall weeks
Temperature_light_mat = cell(1, nstns);
Temperature_heavy_mat = cell(1, nstns);
for n = 1 : nstns
    Temperature_new = Temperature_transformed(:, n);
    RH_light_week = weeks(RH_light_week_index(:, n));
    RH_heavy_week = weeks(RH_heavy_week_index(:, n));
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

    Temperature_light_mat{n} = cell2mat(Temperature_light_cell);
    Temperature_heavy_mat{n} = cell2mat(Temperature_heavy_cell);
end

%% remove empty cells
Temperature_light = Temperature_light_mat(~cellfun('isempty',Temperature_light_mat));
Temperature_heavy = Temperature_heavy_mat(~cellfun('isempty',Temperature_heavy_mat));

%% fit the temporal gaussian process and the warping distribution
distname = "Gamma";
[pd_light, hyp_light] = fitTemporal(Temperature_light, distname);
[pd_heavy, hyp_heavy] = fitTemporal(Temperature_heavy, distname);


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
covfunc0 = {@covMaterniso, 5}; hyp0.cov = hyp_light.cov;
pd0 = pd_light;

% H1
meanfunc1 = @meanConst; 
covfunc1 = {@covMaterniso, 5}; hyp1.cov = hyp_light.cov;
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
figOpt = true;
alpha = 0.1;

% run WGPLRT
[tp,fp,optLogGamma] = FuncWGPLRT(H0, H1, T, M, snP,alpha, printOpt,figOpt);




