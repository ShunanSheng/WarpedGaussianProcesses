function [hyp0, hyp1] = FitTemporalProcess(figOpt)
% given dataset, fit the temporal processes
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
% distname = "Gamma";
distname = "InverseGaussian";
[pd_light, hyp_light] = fitTemporal(Temperature_light, distname);
[pd_heavy, hyp_heavy] = fitTemporal(Temperature_heavy, distname);

%% construct the hypotheses
hyp0 = struct('mean', 0 ,'cov',hyp_light.cov,'dist',pd_light);
hyp1 = struct('mean', 0 ,'cov',hyp_heavy.cov,'dist',pd_heavy);

%% save parameters
save("Experiment/SemiSyntheticExperiment/temporal_hyper.mat","hyp0","hyp1")


%% plot the warping distribution and the histogram
if figOpt == true
    close
    Temperatue_light_flattened = cat(1,Temperature_light{:});
    Temperatue_heavy_flattened = cat(1,Temperature_heavy{:});
    
    file = load('Experiment/SemiSyntheticExperiment/temporal_hyper.mat');
    pd_light = file.hyp0.dist;
    pd_heavy = file.hyp1.dist;

    x = linspace(0, 20, 100);
%     fig1 = tight_subplot(1,1,[.02 .08],[.09 .09],[.08 .02]);
%     qqplot(fig1, Temperatue_light_flattened)
%     grid on 
%     title("")
% %     title("Q-Q plot of Temperature when the average weekly RH is below median",'FontSize',15)
%     xlabel('Standard Normal Quantiles','FontSize',15)
%     ylabel('Quantiles of Input Sample','FontSize',15)
% 
    fig1 = tight_subplot(1,1,[.02 .08],[.09 .09],[.08 .02]);
    qqplot(Temperatue_light_flattened, pd_light)
    grid on
    title("")
%     title("Q-Q plot of Temperature when the average weekly RH is below median",'FontSize',15)
    xlabel('Gamma Quantiles','FontSize',15)
    ylabel('Quantiles of Input Sample','FontSize',15)
    
%     fig2 = tight_subplot(1,1,[.02 .08],[.09 .09],[.08 .02]);
%     qqplot(Temperatue_heavy_flattened, pd_heavy)
%     grid on 
%     xticks([4:2:18])
%     title("")
% %     title("Q-Q plot of Temperature when the average weekly RH is above median",'FontSize',15)  
%     xlabel('Gamma Quantiles','FontSize',15)
%     ylabel('Quantiles of Input Sample','FontSize',15)

%     fig2 = tight_subplot(1,1,[.02 .08],[.09 .09],[.08 .02]);
%     qqplot(Temperatue_heavy_flattened)
%     grid on
%     title("")
% %     title("Q-Q plot of Centered Temperature when the average weekly RH is above median",'FontSize',15)  
%     xlabel('Standard Normal Quantiles','FontSize',15)
%     ylabel('Quantiles of Input Sample','FontSize',15)
% 
%     fig3 = tight_subplot(1,1,[.02 .08],[.09 .09],[.08 .02]);
%     p_heavy = pdf(pd_heavy,x);
%     histogram(10 - Temperatue_heavy_flattened,'Normalization','pdf')
%     hold on
%     grid on 
%     plot(10 - x,p_heavy,'LineWidth',1.5)
% %     title("Histogram of Temperature when the average weekly RH is above median",'FontSize',15)
%     xlabel("Centered Temperature (Celsius)","FontSize",15)
%     ylabel("Frequency","FontSize",15)
    
%     fig4 = tight_subplot(1,1,[.02 .08],[.09 .09],[.08 .02]);
%     p_light = pdf(pd_light,x);
%     histogram(10 - Temperatue_light_flattened,'Normalization','pdf')
%     hold on
%     grid on
%     plot(10 - x,p_light,'LineWidth',1.5)
% %     title("Histogram of Centered Temperature when the average weekly RH is below median ",'FontSize',15)
%     xlabel("Centered Temperature (Celsius)","FontSize",15)
%     ylabel("Frequency","FontSize",15)
end

end