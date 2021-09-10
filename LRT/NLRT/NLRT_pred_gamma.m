function yhat=NLRT_pred_gamma(Lambda,logGamma)
       % Given logGamma and the test statistic Lambda, compute the
       % prediction yhat
       % Input:
       % Lambda : the test statistic
       % logGamma: LRT thereshold
       % Output: yhat
       yhat=log(Lambda)<logGamma;

end