function Ypred=SBLUE(covfunc,hypcov,Yhat,Xtrain,xstar,A,q)
    % Perform SBLUE given the decisions Yhat at the prediction location xstar 
    %
    % Input : 
    % covfunc : the cov function of spatial field
    % hypcov : parameter of cov function 
    % Yhat : the noisy labels
    % A : transition matrix
    % q : the threshold of binary Spatial field
    % Xtrain: the location of existing sensors
    % xstar : the location of precidtion
    %
    % Output:
    % Ypred : the prediction at xstar
    
    c=norminv(1-q);
    N=size(Yhat,1);
    
    mY=meanY(A,c,N);
    Cov_Y=covY(covfunc,hypcov,c,A,Xtrain);
    Cov_xstar=feval(covfunc{:}, hypcov, xstar, Xtrain);
    Cov_g=covgY(A,Cov_xstar,c);
    
    
    %prediction
    g_star_pred=Cov_g / (Cov_Y)*(Yhat-mY);
    Ypred=g_star_pred>c;
end