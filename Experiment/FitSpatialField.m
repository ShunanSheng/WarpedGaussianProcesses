
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
Data_mean = sum(Data_ave_week, 1)/nweeks;
Data_ave_rm = Data_ave_week - Data_mean;

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
stns_loc = horzcat(long, lat);


%% fit the spatial field
% set the prior for the Gaussian spatial field
meanfunc = @meanConst; hyp.mean = sum(Data_mean)/21;
ell = 1/2; sf = 1/2;  
covfunc = {@covMaterniso, 5};
hyp.cov = log([ell;sf]);
likfunc = @likGauss; 
sn = 0.00001; hyp.lik = log(sn); % does setting sn to be small applicable ? 

% setting the prior for the latent gaussian process
prior.mean = {{@priorDelta}};
prior.cov{2}  = {@priorLaplace,mu,s2};
inf = {@infPrior,@infGaussLik,prior};

inf = @infGaussLik;
% X = [stns_loc(:,1)-103.5, stns_loc(:,2)-1] * 100;
X = stns_loc;

%% optimize the hyperparameters for each of nweeks
hyp_lst.logell = zeros(nweeks, 1);
hyp_lst.logsf = zeros(nweeks, 1);
hyp_lst.lik = zeros(nweeks, 1);

for i = 1:nweeks
    hyp2 = minimize(hyp, @gp, -500, inf, meanfunc, covfunc, likfunc, X , Data_ave_rm(i,:)');
    nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, stns_loc, Data_ave_week(i,:)')
    hyp_lst.logell(i) = hyp2.cov(1);
    hyp_lst.logsf(i) = hyp2.cov(2);
    hyp_lst.lik(i) = hyp2.lik; 
end

% choose median over 13 weeks
hyp_final.mean = sum(Data_mean)/21;
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
    
    figure()
    qqplot(reshape(Data_ave_week,[],1))
    title("Q-Q plot of average weekly Relative Humidty")
    
    figure()
    histogram(reshape(Data_ave_week,[],1))
    title("Histogram of average weekly Relative Humidty")
end

end