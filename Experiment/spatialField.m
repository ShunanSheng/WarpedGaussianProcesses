%%% fit the spatial field
%% load data 
clc, clear all, close all
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
targetColumn=3;                              % don't bury magic numbers, use input() to set, maybe...
rootdir = 'Data/Weather_Data_NEA';            % set the root directory location (could use uigetfile)
dDir=dir(rootdir);                            % create a list of files
d=dir(fullfile(rootdir,'*.csv'));  % and pick up all the .csv files therein...

%% choose data from 2012, no missing data
Data_cell = cell(1,numel(d));
stn_lst = cell(numel(d),1);
for i = 1:numel(d)
    stn_name = extractBetween(d(i).name, "AWS","Hrly");
    stn_lst{i} = stn_name{1};
    data = readtable(fullfile(rootdir,d(i).name));  % read the jth file in ith directory
    data_filtered = data(data.Month >=5 &  data.Month <=7 & data.Year==2012 &...
    data.Hour>=5 & data.Hour<=23, :);
    data_RH = data_filtered.RH___;
    if iscell(data_RH)
        data_RH = str2double(data_RH);
    end
    if sum(isnan(data_RH))==1
        disp(i)
    end
    Data_cell{i} = data_RH;   
end
% create data table
Data = [];
for i = 1:21
    Data = horzcat(Data,Data_cell{i});
end
%% calculate average weekly RH
nweeks = floor(1748/19/7);
Data_weeks = Data(1:nweeks*19*7, :);
Data_reshaped = reshape(Data_weeks, 19*7, [],21);
Data_ave_week = reshape(sum(Data_reshaped, 1)./19./7,[],21);

%% fit for spatial field
% plot the stations
% import longtitude and latititude of stations
locations = readtable('Data/Weather_Data_NEA/Stns Metadata.xlsx','VariableNamingRule','modify');
stn_lst = strtrim(stn_lst);
stns_data = locations(ismember(locations.StnNo, stn_lst),:);
%% convert the latitude and longtitude
lat_lst = stns_data.Lat;
lat_rep = str2double(strrep(split(lat_lst,'°'),"'",''));
lat = lat_rep(:,1) + lat_rep(:,2)./60;
long_lst = stns_data.Long;
long_rep = str2double(strrep(split(long_lst,'°'),"'",''));
long = long_rep(:,1) + long_rep(:,2)./60;

%% plot
geoplot([lat(:)],[long(:)],'^')
geobasemap streets-light

%% fit the spatial field




%% test

l = Data_weeks(:,2);
l_reshaped =reshape(l,19*7,[]);
l_ave = (sum(l_reshaped,1)./19./7)';
