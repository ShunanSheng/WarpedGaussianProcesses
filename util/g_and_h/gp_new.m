function [Z, dZ] = gp_new(hyp, inf, meanfunc, covfunc, likfunc, X, GP_temp)
        % new gp function for cell
        n = length(GP_temp);
        Z = 0; 
        dZ.mean = zeros(size(hyp.mean));
        dZ.cov = zeros(size(hyp.cov));
        dZ.lik = zeros(size(hyp.lik));
        for i = 1:n
            [nlZ, dnlZ] = gp(hyp, inf, meanfunc, covfunc, likfunc, X{i}, GP_temp{i});
            Z = Z + nlZ;
            dZ.mean = dZ.mean + dnlZ.mean;
            dZ.cov = dZ.cov + dnlZ.cov;
            dZ.lik = dZ.lik + dnlZ.lik;
        end
end
    