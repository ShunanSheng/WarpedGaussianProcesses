function [Lambda,yhat]=NLRT_pred(D0,D1,delta,logGamma)
    % Given the distance tolerance delta and LRT threshold, compute yhat
    % Input :
    % D0, D1 : the distance matrix between ZI0/ZI1 and ZI
    % delta: distance tolerance
    % logGamma: LRT thereshold
    % Output: yhat
  
    
    nI=size(D0,2);
    nhat=10;
    nbatch=ceil(nI/nhat);
    n0=cell(1,nbatch);n1=cell(1,nbatch);
    
    for k=1:nbatch
        n0{k}=sum(D0(:,(k-1)*nhat+1:k*nhat)<delta,1);
        n1{k}=sum(D1(:,(k-1)*nhat+1:k*nhat)<delta,1);
    end
    N0=horzcat(n0{:});
    N1=horzcat(n1{:});
    Lambda=(N0./N1)'; % the test statistic
    yhat=log(Lambda)<logGamma;
end