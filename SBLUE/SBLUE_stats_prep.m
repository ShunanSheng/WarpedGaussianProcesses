function SBLUEprep=SBLUE_stats_prep(covfunc,hypcov,Xtrain,xstar,q)
        % Compute the variables in SBLUE without knowlegde of the confusion
        % matrix A
        %
        % Input : 
        % covfunc : the cov function of spatial field
        % hypcov : parameter of cov function 
        % q : the threshold of binary Spatial field
        % Xtrain: the location of existing sensors
        % xstar : the location of precidtion
        % Outpute : SBLUE prep
    
        c=norminv(1-q);
        N=size(Xtrain,1);
        K=feval(covfunc{:},hypcov,Xtrain); % Cov_Xtrain
        Cov_xstar=feval(covfunc{:}, hypcov, xstar, Xtrain); % Cov(xsatr, Xtrain)
 
S
        SBLUEprep.Cov_xstar=Cov_xstar;
        SBLUEprep.CovP=struct("P1",P1,"P2",P2,"P3",P3,"P4",P4);
end