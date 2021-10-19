function sz=summaryAuto(z,lag)
    % Summary Statistic 4-dimensional vector recording the autocorrelation 
    % acf(4,:)
    ncol=size(z,2);
    acf=zeros(lag+1,ncol);
    for col=1:ncol
        acf(:,col) = autocorr(z(:,col),lag);
    end
    sz=acf(2:lag+1,:);
end