function Lambda=NLRT_pred_delta(D0,D1,delta)
    % Given the distance tolerance delta, compute the test statistic Lambda
    % Input :
    % D0, D1 : the distance matrix between ZI0/ZI1 and ZI
    % delta: distance tolerance
    % Output: Lambda
  
    
    nI=size(D0,2);
    nhat=1000;
    nbatch=ceil(nI/nhat);
    n0=cell(1,nbatch);n1=cell(1,nbatch);
    
    for k=1:nbatch
        if k*nhat<=nI
            n0{k}=sum(D0(:,(k-1)*nhat+1:k*nhat)<delta,1);
            n1{k}=sum(D1(:,(k-1)*nhat+1:k*nhat)<delta,1);
        else
            n0{k}=sum(D0(:,(k-1)*nhat+1:end)<delta,1);
            n1{k}=sum(D1(:,(k-1)*nhat+1:end)<delta,1);
        end
    end
    N0=horzcat(n0{:});
    N1=horzcat(n1{:});
    
    epsilon=0.1; % To avoid ill-division
    
    Lambda=((N0+epsilon)./(N1+epsilon))'; % the test statistic
    Lambda(isnan(Lambda))=0;

end