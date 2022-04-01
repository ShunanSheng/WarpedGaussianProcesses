function R = SBLUE_risk(Cov_gstar, SBLUE)
    % Compute Bayes risk of SBLUE
    R = diag(Cov_gstar)-diag(SBLUE.Covg/(SBLUE.CovY) * SBLUE.Covg');
end