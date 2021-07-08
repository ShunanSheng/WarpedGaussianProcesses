function f=SimGP(hyp,meanfunc,covfunc,x)
    % Simulate Gaussian Process at points x 
    %
    % Input: meanfunc, covfunc, hyp, x
    % Output: f
    n= size(x,1);
    C = feval(covfunc{:}, hyp.cov, x);
    mu = meanfunc( hyp.mean, x);
    f = chol(C+1e-9*eye(n))'*randn(n, 1) + mu ;
end