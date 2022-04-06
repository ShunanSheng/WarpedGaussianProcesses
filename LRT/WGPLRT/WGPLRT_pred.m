function yhat=WGPLRT_pred(ZP,LRT,logGamma)
    % Compute the predictor given the real data zP
    % For zP of large size, we evaluate the LRT in batches to avoid memory
    % overflow
    % Inputs:
    % zP: point observations
    % LRT: the parameters from WGPLRT_opt.m
    % gamma: LRT thereshold
    %
    % Ouput: yhat
    
    nhat=10000; % The batch size 
    n=size(ZP,2);
    nbatch=ceil(n/nhat); % The number of batches
    
    Yhat=cell(1,nbatch);
    for k=1:nbatch
        if k*nhat<=n
            zP=ZP(:,(k-1)*nhat+1:k*nhat);
        else
            zP=ZP(:,(k-1)*nhat+1:end);
            
        end
        % Evaluatet the test statistic
        Lambda= (LRT.const + sum((LRT.Cinv0 * (zP - LRT.vhat0)) .^ 2) ...
            - sum((LRT.Cinv1 * (zP - LRT.vhat1)) .^ 2)) / 2;
        % Decision
        Yhat{k}=(Lambda>-logGamma)';
    end
    yhat = vertcat(Yhat{:});
 
end