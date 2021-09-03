function Lambda=testStats(zP,LRT)
    % Calcualte the test statistic as in Eqn. 14
    %
    % Input: 
    % zP   : the point observations
    % LRT  : WGPLRT constants
    %
    % Output: 
    % Lambda: the value of the test statistic
    
    
    % Evaluatet the test statistic
    Lambda= (LRT.const + sum((LRT.Cinv0 * (zP - LRT.vhat0)) .^ 2) ...
        - sum((LRT.Cinv1 * (zP - LRT.vhat1)) .^ 2)) / 2;
    
end