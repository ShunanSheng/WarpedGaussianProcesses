function f=SimGP(hyp,meanfunc,covfunc,x)
    n= size(x,1);
    K = covfunc{:}( hyp.cov, x);
    mu = meanfunc( hyp.mean, x);
    f = chol(K+1e-9*eye(n))'*randn(n, 1) + mu ;
end