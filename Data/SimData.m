function [ZP,ZI,y]=SimData(SP,H0,H1)
    % Generate a test dataset 
    %
    % Input: 
    % SP: paramaters for the spatial process
    % H0,H1 : paramters for null/alternative hypotheses
    % Output: ZP,ZI,y

    % Binary spatial field GP(0,C)
    meanfunc = SP.meanfunc; 
    covfunc = {SP.covfunc};
    hyp=SP.hyp;

    % Location of sensors 
    n = 5; xinf=-10; xsup=10;
    [X,Y]= meshgrid(linspace(xinf,xsup,n),linspace(xinf,xsup,n));
    xSp=reshape(X,[],1);
    ySp=reshape(Y,[],1); 

    % Generate the lantent binary spatial field
    warpfunc=@(pd,p) invCdf(pd,p);
    y=SimWGP(hyp,meanfunc,covfunc,warpfunc,[xSp,ySp])

    % Total number of sensors
    N=length(y); 
    % The indexes of point sensors and integral sensors
    xI=1:5:n^2;NI=length(xI);
    xP=setdiff(1:n^2,xI);NP=length(xP);
    % without loss of generality we may assume M=K in practice
    M=20; K=20; snP=0.1; snI=0.1;

    % Hypotheses for Temproal Process
    meanfunc0 = H0.meanfunc; 
    covfunc0 = {H0.covfunc};
    hyp0=H0.hyp;

    meanfunc1 = H1.meanfunc; 
    covfunc1 = {H1.covfunc};
    hyp1=H1.hyp;

    % The Point Observations
    ZP=zeros(M,NP);t=linspace(0,hyp0.t,M)';
    for i=1:NP
       ZP(:,i)= SimPtData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,snP,y(xP(i))); 
    end
    % The Integral Observations
    ZI=zeros(K,NI);
    for i=1:NI
       ZI(:,i)= SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,y(xI(i))); 
    end

end