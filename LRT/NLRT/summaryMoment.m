function sz=summaryMoment(z)
    % Summary Statistic 4-dimensional vector 
    % (mean,var,mode,kurtosis)
    sz=zeros(4,1);
    sz(1)=mean(z);
    sz(2)=var(z);
    sz(3)=mode(z);
    sz(4)=kurtosis(z);
end