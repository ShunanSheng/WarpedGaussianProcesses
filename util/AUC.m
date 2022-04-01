function auc = AUC(TP, FP)
    % Compute Area Under Curve (AUC) given the False positive rate (FP) and True
    % positive rate (FP)
    if size(TP, 1) > 1 
        TP = [TP; 1]; % add the (1,1) point, in case the FPR does not reach 1
        FP = [FP; 1];
    else
        TP = [TP, 1];
        FP = [FP, 1];
    end
    dx = diff(FP);
    h = 1/2 .* (TP(1:end-1) + TP(2:end));
    auc = sum(dx .* h);
end