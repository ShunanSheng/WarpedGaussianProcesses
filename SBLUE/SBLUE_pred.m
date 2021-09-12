function Ypred=SBLUE_pred(SBLUE,Yhat)
   % Input: 
   % Yhat : the noisy labels
   % SBLUE : the SBLUE parameters
   % Output: Ypred : the prediction at xstar

   % prediction
    g_star_pred=SBLUE.Covg / (SBLUE.CovY)*(Yhat-SBLUE.mY);
    Ypred=g_star_pred>SBLUE.c;
end