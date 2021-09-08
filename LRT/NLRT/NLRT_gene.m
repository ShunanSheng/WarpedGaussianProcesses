function [ZI0,ZI1]=NLRT_gene(hyp0,C0,mu0,hyp1,C1,mu1, warpfunc,K,kw,snI,J)
    % Generate J samples from null hypothesis (ZI0) and J samples from
    % altenative hypothesis (ZI1)
    
    ZI0=SimIntData(hyp0,C0,mu0, warpfunc,K,kw,snI,J); % Generate J samples from each hypothesis
    ZI1=SimIntData(hyp1,C1,mu1, warpfunc,K,kw,snI,J);
    
end