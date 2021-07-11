function yhat=WGPLRT(zP,H0,H1,warpinv,t,snP,logGamma)
    % Conduct Warped Gaussian Process LRT given the point observations
    %
    % Input: 
    % zP   : the point observations
    % H0,H1: parameters for null/alternative hypotheses
    % warpinv : the inverse warping function handle G=warpinv(pd,x)
    % t    : the time points
    % snP  : noise for point observation
    % gamma: LRT thereshold
    %
    % Output: 
    % yhat : the decision

    n=size(t,1);x0=ones(n,1)*0.1;
    hyp0=H0.hyp;hyp1=H1.hyp;
    lb0=hyp0.lb;ub0=hyp0.ub;
    lb1=hyp1.lb;ub1=hyp1.ub;
    
    pd0=hyp0.dist;pd1=hyp1.dist;
    covfunc0 = {H0.covfunc};
    covfunc1 = {H1.covfunc};
    
    % Evalaute K0,K1
    K0=feval(covfunc0{:}, hyp0.cov, t);
    K1=feval(covfunc1{:}, hyp1.cov, t);
    Kchol0=chol(K0+1e-9*eye(n)); 
    Kichol0=Kchol0\eye(n); 
    Klogdet0=2*trace(log(Kchol0));
    Kinv0=Kichol0*Kichol0';
    
    Kchol1=chol(K1+1e-9*eye(n)); 
    Kichol1=Kchol1\eye(n); 
    Klogdet1=2*trace(log(Kchol1));
    Kinv1=Kichol1*Kichol1';
    
    % Perform Laplace approximation get vhat, A
    [Qval0,vhat0,A0]=LaplaceApproximation(pd0,Kinv0,warpinv,x0,lb0,ub0);
    [Qval1,vhat1,A1]=LaplaceApproximation(pd1,Kinv1,warpinv,x0,lb1,ub1);
    

%     try chol(A0);
%         disp('Matrix is symmetric positive definite.')
%     catch ME
%         disp('Matrix is not symmetric positive definite')
%     end
    
    % Compute the test statistic
    Lambda=testStats(A0,vhat0,Qval0,Klogdet0,A1,vhat1,Qval1,Klogdet1,snP,zP);
    
    % Decision
    yhat=Lambda>-logGamma;
    
end