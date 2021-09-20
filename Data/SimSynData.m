function Data=SimSynData(SP,H0,H1,warpfunc,modelHyp)
    % Generate a test dataset 
    %
    % Input: 
    % SP : paramaters for the spatial process
    % H0,H1 : paramters for null/alternative hypotheses
    % modelHyp : parameters for sensor network (T,M,K,snI,snP)
    % warpfunc : warpfunc handle
    
    % Output: 
    % ZP : all point observations M x NP 
    % ZI : all integral observations K x NI
    % y  : ground truth over the spatial field
    % xP : index of the point sensors
    % xI : index of the integral sensors
    % x  : all locations
    
    
    % Binary spatial field GP(0,C)
    meanfunc = SP.meanfunc; 
    covfunc = {SP.covfunc};
    hyp=SP.hyp;

    % Location of sensors 
    n = 50; xinf=-5; xsup=5;
    [X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
    xSp=reshape(X,[],1);
    ySp=reshape(Y,[],1); 
    x=[xSp,ySp];
    
    % Generate the lantent binary spatial field
    y=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);
    
    % Total number of sensors
    N=length(y); 
    
    % The index of training and test data
    indexTest=(1:5:N)';
    indexTrain=setdiff(1:N,indexTest)';
    
    % The indexes of point sensors and integral sensors
    xI=indexTrain(1:2:end);xI0=xI(y(xI)==0);
    xI1=xI(y(xI)==1);nI0=length(xI0);nI1=length(xI1);NI=length(xI);
    xP=setdiff(indexTrain,xI);xP0=xP(y(xP)==0);
    xP1=xP(y(xP)==1);nP0=length(xP0);nP1=length(xP1);NP=length(xP);
    
    
    % without loss of generality we may assume M=K in practice
    T=modelHyp.T;M=modelHyp.M; K=modelHyp.K; snI=modelHyp.snI; snP=modelHyp.snP;

    % Hypotheses for Temproal Process
    meanfunc0 = H0.meanfunc; 
    covfunc0 = {H0.covfunc};
    hyp0=H0.hyp;

    meanfunc1 = H1.meanfunc; 
    covfunc1 = {H1.covfunc};
    hyp1=H1.hyp;
    

    % The Point Observations
    t=linspace(0,T,M)'; % the time points to observe the point observations
    % parameters
    CP0 = chol(feval(covfunc0{:}, hyp0.cov, t)+1e-9*eye(M));
    muP0 = meanfunc0( hyp0.mean, t);
    CP1 = chol(feval(covfunc1{:}, hyp1.cov, t)+1e-9*eye(M));
    muP1 = meanfunc1( hyp1.mean, t);
    
        
    ZP0=SimPtData(hyp0,CP0,muP0,warpfunc,t,snP,nP0);
    ZP1=SimPtData(hyp1,CP1,muP1,warpfunc,t,snP,nP1);
    
    % The Integral Observations
    kw= ceil(exp(log(10000*T/K/180)/4)); % calculate the number of point neeed per window under Simpson's rule with 0.01 error
    kw= round(kw/2)*2;n=kw*K;tI=linspace(0,T,n)';

    CI0 = chol(feval(covfunc0{:}, hyp0.cov, tI)+1e-9*eye(n));
    muI0 = meanfunc0( hyp0.mean, tI);

    CI1 = chol(feval(covfunc1{:}, hyp1.cov, tI)+1e-9*eye(n));
    muI1 = meanfunc1( hyp1.mean, tI);
    
    ZI0=SimIntData(hyp0,CI0,muI0, warpfunc,K,kw,snI,nI0);
    ZI1=SimIntData(hyp1,CI1,muI1, warpfunc,K,kw,snI,nI1);
    
    % Create Data structure
    Data.ZP.H0=ZP0;
    Data.ZP.H1=ZP1;
    Data.ZI.H0=ZI0;
    Data.ZI.H1=ZI1;
    Data.indexTrain=indexTrain;
    Data.indexTest=indexTest;
    Data.x=x;
    Data.y=y;
    Data.xI.H0=xI0;
    Data.xI.H1=xI1;
    Data.xP.H0=xP0;
    Data.xP.H1=xP1;
    Data.t=t;
    
end