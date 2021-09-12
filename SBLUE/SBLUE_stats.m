function SBLUE=SBLUE_stats(SBLUEprep,A,q)
    % Compute the variables in SBLUE with knowlegde of the confusion matrix
    %
    % Input : 
    % covfunc : the cov function of spatial field
    % hypcov : parameter of cov function 
    % A : transition matrix
    % q : the threshold of binary Spatial field
    % Xtrain: the location of existing sensors
    % xstar : the location of precidtion
    %
    % Output: SBLUE
    
    
    c=norminv(1-q);
    CovP=SBLUEprep.CovP;
    Cov_xstar=SBLUEprep.Cov_xstar;
    
    N=size(Cov_xstar,2);
    
    % compute parameters
    mY=meanY(A,c,N);
%     Cov_xtrain=feval(covfunc{:},hypcov,Xtrain);
        
    Cov_Y=covY(A,CovP,mY,c);
    Cov_g=covgY(A,Cov_xstar,c);
    
    SBLUE.mY=mY;
    SBLUE.CovY=Cov_Y;
    SBLUE.Covg=Cov_g;
    SBLUE.c=c;
 
end