function temp_process_new = processTemp(temp_process)
%%% remove linear trend from the time series and flip the left skewed data
%%% by (40 - data)
     alpha = 0.05;
     n = length(temp_process);
     [H,~] = Mann_Kendall(temp_process,alpha);
     if H==1
         b = polyfit(1:n, temp_process, 1);
         y = (1:n)'* b(1);
         temp_process_new = temp_process - y; 
     end
     temp_process_new = 10 - temp_process_new;
     [H2,~] = Mann_Kendall(temp_process_new,alpha);
     
     if H2 ==1
         warning("there is linear trend after processing")
     end
end