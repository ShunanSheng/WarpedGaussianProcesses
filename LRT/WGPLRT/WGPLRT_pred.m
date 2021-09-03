% Improved version
function yhat=WGPLRT_pred(zP,LRT,logGamma)

    % Compute the predictor given the real data zP
    % Inputs:
    % zP: point observations
    % LRT: the parameters from WGPLRT_opt.m
    % gamma: LRT thereshold
    %
    % Ouput: yhat
    
    
    % Evaluatet the test statistic
    Lambda= (LRT.const + sum((LRT.Cinv0 * (zP - LRT.vhat0)) .^ 2) ...
        - sum((LRT.Cinv1 * (zP - LRT.vhat1)) .^ 2)) / 2;

    % Decision
    yhat=Lambda>-logGamma;
    
    
end