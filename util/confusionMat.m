function [TP,FP]=confusionMat(y,yhat)
    % Compute the confusion matrix given the ground truth y (0,1) (vector) and predictions
    % yhat (0,1) (vector), let's say 1 indictaes true and 0 indicate false
    % Input : 
    % y : ground truth
    % yhat : predictions
    
    % Output :
    % TP,FP, true/false positive rate
    
    
    tp=sum((y==1)&(yhat==1));
    fp=sum((y==0)&(yhat==1));    
    fn=sum((y==1)&(yhat==0));
    tn=sum((y==0)&(yhat==0));
    
    TP=tp/(tp+fn);
    FP=fp/(fp+tn);
end