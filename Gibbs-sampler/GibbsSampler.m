function g_samples = GibbsSampler(p, mu, Sigma, c, burnin, S)
    % Given the p = 1/(1 + Lambda), mu , Sigma, c, burin number, sampling
    % number, the following algrithm performs Gibbs sampler to approximate the
    % latent spatial field g

    N = length(p);
    g_old = chol(Sigma+1e-9*eye(N))'*randn(N, 1) + mu;
    pd = makedist("Normal");
    % burn-in stage, run the markov chain until mixing
    for iter = 1:burnin
        for n = 1:N
            a = binornd(1,p(n));
            C_n = Sigma(n, [1:n-1,n+1:end]);
            C_n_n = Sigma([1:n-1,n+1:end],[1:n-1,n+1:end]);
            g_n = g_old([1:n-1,n+1:end]);
            mu_n = mu([1:n-1,n+1:end]);
            mun = mu(n) + C_n * ( C_n_n\(g_n - mu_n));
            sigman = sqrt(Sigma(n,n) - C_n * (C_n_n\C_n'));
            cn = (c - mun)/sigman;
            if a == 1
                t = truncate(pd, min(cn, 5),Inf);
            else
                t = truncate(pd, -Inf, max(cn, -5));
            end
            g_old(n) = random(t);
        end
    end
    
    % draw new samples from the chain
    g_samples = zeros(N, S);
    for n = 1:N
        a = binornd(1,p(n), S, 1);
        C_n = Sigma(n, [1:n-1,n+1:end]);
        C_n_n = Sigma([1:n-1,n+1:end],[1:n-1,n+1:end]);
        g_n = g_old([1:n-1,n+1:end]);
        mu_n = mu([1:n-1,n+1:end]);
        mun = mu(n) + C_n * ( C_n_n\(g_n - mu_n));
        sigman = sqrt(Sigma(n,n) - C_n * (C_n_n\C_n'));
        cn = (c - mun)/sigman;
        gns0 = random(truncate(pd, -Inf, max(cn, -5)), S, 1);
        gns1 = random(truncate(pd, min(cn, 5),Inf), S, 1);
        g_samples(n, :) = ((1 - a).* gns0 + a.* gns1)';
    end
end


