function [pd, hyp2, nlml] = fitTemporal(temp_process, distname)
    %%% Given the temporal data and the marginal distribution, find the
    %%% hyperparameters for the distribution and the latent gaussain
    %%% process
    %%% Input: 
    %%% temp_process
    %%% distname : the name of warping distribution
    %%% Output:
    %%% pd : the warping distribution
    %%% hyp2 : the hyperparameters of the latent GP
    %%% nlml : the negative log marginal likelihood
    
    pd = fitdist(temp_process, distname);
    gp_temp = norminv(cdf(pd, temp_process));
    x = (1:length(gp_temp))';

    meanfunc = @meanConst; hyp.mean = 0;
    ell = 1; sf = 1;  
    covfunc = {@covMaterniso, 3};
    hyp.cov = log([ell;sf]);
    likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
    
    % setting the prior for the latent gaussian process
    prior.mean = {{@priorDelta}};
    prior.lik = {{@priorDelta}};
    prior.cov = {[];{@priorDelta}};
    inf = {@infPrior,@infGaussLik,prior};

    hyp2 = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x, gp_temp);
    nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, gp_temp)
    
end