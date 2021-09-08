function yhat=NLRT_pred(D0,D1,delta,logGamma)
    % Given the distance tolerance delta and LRT threshold, generate ZP
    % Input :
    % D0, D1 : the distance matrix between ZI0/ZI1 and ZI
    % delta: distance tolerance
    % logGamma: LRT thereshold
    % Output: yhat
  
    n0=sum(D0<delta,1)';n1=sum(D1<delta,1)';
    Lambda=n0./n1;
    yhat=log(Lambda)<logGamma;
end