function SBLUE=SBLUE_stats(SBLUEprep,A,q)
    % Compute the mean, covariance in SBLUE with knowlegde of the confusion matrix
    %
    % Input : 
    % SBLUEprep: the SBLUE statistics without knowing the confusion matrix
    % A
    % A : confusion matrix
    % q : the threshold of binary Spatial field
    %
    % Output: SBLUE
    
    
    c=norminv(1-q);
    CovP=SBLUEprep.CovP;
    Cov_xstar=SBLUEprep.Cov_xstar;
    
    N=size(Cov_xstar,2);
    
    % compute parameters
    mY=meanY(A,c,N);        
    Cov_Y=covY(A,CovP,mY,c);
    Cov_g=covgY(A,Cov_xstar,c);
    
    % create strtucture
    SBLUE.mY=mY;
    SBLUE.CovY=Cov_Y;
    SBLUE.Covg=Cov_g;
    SBLUE.c=c;
 
end