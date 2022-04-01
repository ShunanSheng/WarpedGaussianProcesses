function result=F1score(y,yhat)
    % Compute the F1 score of the binary classification
    % Input : 
    % y : ground truth
    % yhat : predictions
    
    % Output :
    % result : the F1 score
    
    y=y(:);
    yhat=yhat(:);
    
    tp=sum((y==1)&(yhat==1));
    fp=sum((y==0)&(yhat==1));    
    fn=sum((y==1)&(yhat==0));
    
    result= 2*tp/(2*tp+fp+fn);

end