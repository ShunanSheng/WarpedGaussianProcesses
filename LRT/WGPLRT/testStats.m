function Lambda=testStats(ZP,LRT)
    % Calcualte the test statistic as in Eqn. 14
    %
    % Input: 
    % ZP   : the point observations
    % LRT  : WGPLRT constants
    %
    % Output: 
    % Lambda: the value of the test statistic -2logLambda(ZP)
    
    
    nhat=1000; % The batch size 
    n=size(ZP,2);
    nbatch=ceil(n/nhat); % The number of batches
    
    lambda=cell(1,nbatch);
    for k=1:nbatch
        if k*nhat<=n
            zP=ZP(:,(k-1)*nhat+1:k*nhat);
        else
            zP=ZP(:,(k-1)*nhat+1:end); 
        end
    
        % Evaluate the test statistic
        lambda{k}= ((LRT.const + sum((LRT.Cinv0 * (zP - LRT.vhat0)) .^ 2) ...
            - sum((LRT.Cinv1 * (zP - LRT.vhat1)) .^ 2)) / 2)';
    end
    Lambda=vertcat(lambda{:});
end