function Ypred=SBLUE(covfunc,hypcov,Yhat,Xtrain,xstar,A,q)
    c=norminv(1-q);
    N=size(Yhat,1);
    
    mY=meanY(A,c,N);
    Cov_Y=covY(covfunc,hypcov,c,A,Xtrain);
    Cov_xstar=feval(covfunc{:}, hypcov, xstar,Xtrain);
    Cov_g=covgY(A,Cov_xstar,c);
    
    
    %prediction
    g_star_pred=Cov_g / (Cov_Y)*(Yhat-mY);
    Ypred=g_star_pred>c;
end