function sz=summaryAutoMoment(z)
    % Summary Statistic 5-dimensional vector 
    % (mean,var,mode,kurtosis, ACF(,1))
    ncol=size(z,2);
    sz=zeros(5,ncol);
    sz(1,:)=mean(z);
    sz(2,:)=var(z);
    sz(3,:)=mode(z);
    sz(4,:)=kurtosis(z);
    
    lag=1;
    acf=zeros(lag+1,ncol);
    for col=1:ncol
        acf(:,col) = autocorr(z(:,col),lag);
    end
    sz(5,:)=acf(2,:);
end