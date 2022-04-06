function [pd, hyp2] = fitTemporal(temp_process, distname)
    %%% Given the temporal data and the marginal distribution, find the
    %%% hyperparameters for the distribution and the latent gaussain
    %%% process
    %%% Input: 
    %%% temp_process
    %%% distname : the name of warping distribution
    %%% Output:
    %%% pd : the warping distribution
    %%% hyp2 : the hyperparameters of the latent GP
    
    meanfunc = @meanConst; hyp.mean = 0;
    ell = 1; sf = 1;  
    covfunc = {@covMaterniso, 5};
    hyp.cov = log([ell;sf]);
    likfunc = @likGauss; sn = 0.01; hyp.lik = log(sn);

    % setting the prior for the latent gaussian process
    prior.mean = {{@priorDelta}};
    prior.cov = {[];{@priorDelta}};
    inf = {@infPrior,@infGaussLik,prior};

    % if the input is a sequence, perform estimation directly
    if isa(temp_process, 'double')
        pd = fitdist(temp_process, distname);
        gp_temp = norminv(cdf(pd, temp_process));
        x = (1:length(gp_temp))';
        
        hyp2 = minimize(hyp, @gp, -500, inf, meanfunc, covfunc, likfunc, x, gp_temp);
        
    % if the input is a cell, define a new gp function for cell, with Z, dZ
    % being sum of all cells
    elseif isa(temp_process, 'cell')
        n = length(temp_process);
        temp_length = cellfun('length', temp_process);
        pd = fitdist(cat(1, temp_process{:}), distname);
        
        GP_temp = cell(n,1);
        X = cell(n, 1);
        for j = 1:n
            X{j} = (1:temp_length(n))';
            GP_temp{j} = norminv(cdf(pd, temp_process{n}));
        end
        hyp2 = minimize(hyp, @gp_new, -100, inf, meanfunc, covfunc, likfunc, X, GP_temp);
        
    else 
        error("Wrong input type")
    end
    
    
end