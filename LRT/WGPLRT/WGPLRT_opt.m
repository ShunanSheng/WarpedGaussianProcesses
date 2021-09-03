% Improved version
function LRT=WGPLRT_opt(H0,H1,warpinv,t,x0,snP)
% function [LA0,LA1]=WGPLRT_opt(H0,H1,warpinv,t,x0)
    
    % Conduct Warped Gaussian Process LRT given the point observations
    %
    % Input: 
    % H0,H1: parameters for null/alternative hypotheses
    % warpinv : the inverse warping function handle G=warpinv(pd,x)
    % t    : the time points
    % x0   : the initial point
    % Output: 
    % LA0,LA1:

    n=size(t,1); % x0 is the initial point
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
    
%     Klogdet0=2*trace(log(Kchol0));
    % better efficiency
    Klogdet0=2*sum(log(diag(Kchol0)));
    
    Kinv0=Kichol0*Kichol0';
    
    Kchol1=chol(K1+1e-9*eye(n)); 
    Kichol1=Kchol1\eye(n); 
    
%     Klogdet1=2*trace(log(Kchol1));
    % better efficiency
    Klogdet1=2*sum(log(diag(Kchol1)));
    Kinv1=Kichol1*Kichol1';
    
    % Perform Laplace approximation get vhat, A
    x00=x0(:,1);
    x01=x0(:,2);
    
    [Qval0,vhat0,A0]=LaplaceApproximation(pd0,Kinv0,warpinv,x00,lb0,ub0);
    [Qval1,vhat1,A1]=LaplaceApproximation(pd1,Kinv1,warpinv,x01,lb1,ub1);
    
    % place all data-independent computation here for better efficiency
    
    [A0_eigvec, A0_eigval] = eig(A0);
    [A1_eigvec, A1_eigval] = eig(A1);
    
    B0logdet = sum(log(diag(A0_eigval) + snP^2));
    B1logdet = sum(log(diag(A1_eigval) + snP^2));
    
    Cinv0 = diag(sqrt(1 ./ ((1 ./ diag(A0_eigval)) + snP^2))) * A0_eigvec';
    Cinv1 = diag(sqrt(1 ./ ((1 ./ diag(A1_eigval)) + snP^2))) * A1_eigvec';
    
    testStat_const = B0logdet+Klogdet0-2*Qval0-B1logdet-Klogdet1+2*Qval1;
    
    LRT = struct('const', testStat_const, 'vhat0', vhat0, ...
        'Cinv0', Cinv0, 'vhat1', vhat1, 'Cinv1', Cinv1);
    
    % to compute the test statistic of the LRT when the observation is zP:
    % testStat = (LRT.const + sum((LRT.Cinv0 * (zP - LRT.vhat0)) .^ 2) ...
    %       - sum((LRT.Cinv1 * (zP - LRT.vhat1)) .^ 2)) / 2;
    
    
%     LA0=struct("Qval",Qval0,"vhat",vhat0,"A",A0,"Klogdet",Klogdet0); % resulting parameters
%     LA1=struct("Qval",Qval1,"vhat",vhat1,"A",A1,"Klogdet",Klogdet1);
    
end