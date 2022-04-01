function [TPR,FPR]=confusionMat(y,yhat)
    % Compute the confusion matrix given the ground truth y (0,1) (vector) and predictions
    % yhat (0,1) (vector), let's say 1 indictaes true and 0 indicate false
    % Input : 
    % y : ground truth
    % yhat : predictions
    
    % Output :
    % TP,FP, true/false positive rate
    
    % Convert inputs to column vectors
    y=y(:);
    yhat=yhat(:);
    
    tp=sum((y==1)&(yhat==1));
    fp=sum((y==0)&(yhat==1));    
    fn=sum((y==1)&(yhat==0));
    tn=sum((y==0)&(yhat==0));
    
    TPR=tp/(tp+fn);
    FPR=fp/(fp+tn);
end