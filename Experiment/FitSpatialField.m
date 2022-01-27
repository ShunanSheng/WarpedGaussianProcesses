
function [hyp_final, stns_loc] = FitSpatialField(figOpt)
%%% fit the spatial field given RH data from 21 stations across Singapore
%% load data 
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
rootdir = 'Data/Weather_Data_NEA';            % set the root directory location (could use uigetfile)
d=dir(fullfile(rootdir,'*.csv'));  % and pick up all the .csv files therein...

%% choose data from 2012, no missing data
nstns = numel(d);
Data_cell = cell(1,nstns);
stns_lst = cell(nstns,1);
for i = 1:nstns
    stn_name = extractBetween(d(i).name, "AWS","Hrly");
    stns_lst{i} = stn_name{1};
    data = readtable(fullfile(rootdir,d(i).name));  % read the jth file in ith directory
    data_filtered = data(data.Month >=6 &  data.Month <=9 & data.Year==2012 &...
    data.Hour>=5 & data.Hour<=23, :);
    data_RH = data_filtered.RH___;
    if iscell(data_RH)
        data_RH = str2double(data_RH);
    end
    if sum(isnan(data_RH))==1
        disp("station", stn_name)
        warning("Missing data")
    end
    Data_cell{i} = data_RH;   
end
% create data table
Data = zeros(size(Data_cell{1},1), nstns);
for i = 1:nstns
    Data(:, i) = Data_cell{i};
end

%% calculate average weekly RH
nweeks = floor(size(Data_cell{1},1)/19/7);
Data_weeks = Data(19*2+1:19*2+nweeks*19*7, :);
Data_reshaped = reshape(Data_weeks, 19*7, [],nstns);
Data_ave_week = reshape(sum(Data_reshaped, 1)./19./7,[],nstns);

%% fit for spatial field
% plot the stations
locations = readtable('Data/Weather_Data_NEA/Stns Metadata.xlsx','VariableNamingRule','modify');
stns_lst = strtrim(stns_lst);
stns_data = locations(ismember(locations.StnNo, stns_lst),:);

%% convert latitude and longtitude
lat_lst = stns_data.Lat;
lat_rep = str2double(strrep(split(lat_lst,'°'),"'",''));
lat = lat_rep(:,1) + lat_rep(:,2)./60;
long_lst = stns_data.Long;
long_rep = str2double(strrep(split(long_lst,'°'),"'",''));
long = long_rep(:,1) + long_rep(:,2)./60;
stns_loc = horzcat(lat, long);

%% fit the spatial field
% set the prior for the Gaussian spatial field
meanfunc = @meanConst; hyp.mean = 0;
ell = 1; sf = 1;  
covfunc = {@covMaterniso, 3};
hyp.cov = log([ell;sf]);
likfunc = @likGauss; 
sn = 0.00001; hyp.lik = log(sn); % does setting sn to be small applicable ? 
inf = @infGaussLik;

%% optimize the hyperparameters for each of nweeks
hyp_lst.mean = zeros(nweeks, 1);
hyp_lst.logell = zeros(nweeks, 1);
hyp_lst.logsf = zeros(nweeks, 1);
hyp_lst.lik = zeros(nweeks, 1);

for i = 1:nweeks
    hyp2 = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, stns_loc, Data_ave_week(i,:)');
%     nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, stns_loc, Data_ave_week(i,:)');
    hyp_lst.mean(i) = hyp2.mean;
    hyp_lst.logell(i) = hyp2.cov(1);
    hyp_lst.logsf(i) = hyp2.cov(2);
    hyp_lst.lik(i) = hyp2.lik; 
end

% choose median over 13 weeks
hyp_final.mean = median(hyp_lst.mean);
hyp_final.cov(1) =median(hyp_lst.logell);
hyp_final.cov(2) = median(hyp_lst.logsf);
hyp_final.lik = median(hyp_lst.lik);
hyp_final.thres = median(reshape(Data_ave_week,[],1));

assert(hyp_final.lik < log(1e-4), "not negligible noise")

%% plot stations
if figOpt == true
    figure()
    geoscatter([lat(:)]+ 0.003,[long(:)]+0.02,'r','^')
    geobasemap streets-light
    text([lat(:)],[long(:)],stns_data.StnNo)
    title("Geographical locations of sensor stations")
end

end