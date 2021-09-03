% Improved version
function yhat=WGPLRT_pred(zP,LRT,logGamma)
% function yhat=WGPLRT_pred(zP,LA0,LA1,snP,logGamma)
    % Compute the test statistic
    % zP: point observations
    % LA0,LA1: the parameters from WGPLRT_opt.m
    % snP  : noise for point observation
    % gamma: LRT thereshold
    %
    % Ouput: yhat
    
    
    % Evaluatet the test statistic
    Lambda= (LRT.const + sum((LRT.Cinv0 * (zP - LRT.vhat0)) .^ 2) ...
        - sum((LRT.Cinv1 * (zP - LRT.vhat1)) .^ 2)) / 2;

    % Decision
    yhat=Lambda>-logGamma;
end