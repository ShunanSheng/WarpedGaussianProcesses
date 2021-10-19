function R=SBLUE_risk(SBLUE)
    % Compute Bayes risk of SBLUE
    R=1-diag(SBLUE.Covg/(SBLUE.CovY) * SBLUE.Covg');
    
end