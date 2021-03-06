function [varargout] =SBLUE_pred(SBLUE,Yhat)
   % Compute SBLUE given Yhat
   % Input: 
   % Yhat : the noisy labels
   % SBLUE : the SBLUE parameters
   
   % Output: Ypred : the prediction at xstar

   % prediction
   g_star_pred = SBLUE.mXstar+SBLUE.Covg / (SBLUE.CovY)*(Yhat-SBLUE.mY);
   Ypred = double(g_star_pred>SBLUE.c);
   varargout{1} = Ypred;
   if nargout > 1
       varargout{2} = g_star_pred;
   end      

end