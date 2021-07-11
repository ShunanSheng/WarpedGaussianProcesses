function [ZP,ZI,y,xP,xI,indexTrain,indexTest,x]=SimSynData(SP,H0,H1,warpfunc,modelHyp)
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
    n = 10; xinf=-10; xsup=10;
    [X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
    xSp=reshape(X,[],1);
    ySp=reshape(Y,[],1); 
    x=[xSp,ySp];
    
    % Generate the lantent binary spatial field
    y=SimWGP(hyp,meanfunc,covfunc,warpfunc,x);
    
    % Total number of sensors
    N=length(y); 
    
    % The index of training and test data
    indexTest=1:5:N;
    indexTrain=setdiff(1:N,indexTest);
    
    
    % The indexes of point sensors and integral sensors
    xI=indexTrain(1:2:end);NI=length(xI);
    xP=setdiff(indexTrain,xI);NP=length(xP);
    
    
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
    ZP=zeros(NP,M);t=linspace(0,hyp0.t,M)';
    for i=1:NP
       ZP(i,:)= SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,y(xP(i))); 
    end
    % The Integral Observations
    ZI=zeros(NI,K);
    for i=1:NI
       ZI(i,:)= SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,y(xI(i))); 
    end

end