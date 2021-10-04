function SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c)
    % Compute the mean, covariance in SBLUE with knowlegde of the transition matrix
    %
    % Input : 
    % SBLUEprep: the SBLUE statistics without knowing the transition matrix
    % transitionMat : the structure of transition "matrices" (only p01,p11 are used)
    % size {Ntrain x 1 , Ntrain x 1}
    %
    % Output: SBLUE
    

    CovP=SBLUEprep.CovP;
    Cov_xstar=SBLUEprep.Cov_xstar;
    chat=SBLUEprep.chat;
    
    
    % compute parameters
    mY=meanY(transitionMat,chat);        
    Cov_Y=covY(transitionMat,CovP,mY,chat);
    Cov_g=covgY(transitionMat,Cov_xstar,chat);
    
    % create strtucture
    SBLUE.mY=mY;
    SBLUE.CovY=Cov_Y;
    SBLUE.Covg=Cov_g;
    SBLUE.mXstar=SBLUEprep.mXstar;
    SBLUE.c=c;
    
 
end


