function yhat=WGPLRT_pred(zP,LA0,LA1,snP,logGamma)
    % Compute the test statistic
    % zP: point observations
    % LA0,LA1: the parameters from WGPLRT_opt.m
    % snP  : noise for point observation
    % gamma: LRT thereshold
    %
    % Ouput: yhat
    
    % Assign the parameters
    Qval0=LA0.Qval;A0=LA0.A;vhat0=LA0.vhat;Klogdet0=LA0.Klogdet;
    
    Qval1=LA1.Qval;A1=LA1.A;vhat1=LA1.vhat;Klogdet1=LA1.Klogdet;
    
    % Evaluatet the test statistic
    Lambda=testStats(A0,vhat0,Qval0,Klogdet0,A1,vhat1,Qval1,Klogdet1,snP,zP);

    % Decision
    yhat=Lambda>-logGamma;
end