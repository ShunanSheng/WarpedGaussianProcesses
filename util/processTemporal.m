function [Temperature_light, Temperature_heavy] = processTemporal(RH_raw, Temperature_raw)
    %% compute the total rainfall over weeks
    RH_reshaped = reshape(RH_raw, 19*7, []);
    RH_total_week = sum(RH_reshaped, 1)./19./7;
    
    %% compute the total rainfall over weeks
    RH_reshaped = reshape(RH_raw, 19*7, []);
    RH_total_week = sum(RH_reshaped, 1)./19./7;

    %% thresholding based on median value
    RH_med = median(RH_total_week);
    % RH_med = 75.6451;
    RH_light_week = weeks(RH_total_week < RH_med);
    RH_heavy_week = weeks(RH_total_week >= RH_med);

    %% process Temperature data
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

end