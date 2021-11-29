function auc = AUC(TP, FP)
    % Compute Area Under Curve (AUC) given the False positive rate (FP) and True
    % positive rate (FP)
    dx = diff(FP);
    h = 1/2 .* (TP(1:end-1) + TP(2:end));
    auc = sum(dx .* h);
end