function yhat=NLRT(zI,H0,H1,warpfunc,K,snI,sumstats,d,delta,gamma)
    meanfunc0 = H0.meanfunc; 
    covfunc0 = {H0.covfunc};
    hyp0=H0.hyp;

    meanfunc1 = H1.meanfunc; 
    covfunc1 = {H1.covfunc};
    hyp1=H1.hyp;
    
    J=100;n0=0;n1=0;y=sumstats(zI);
   % Generating data
    for j=1:J
        z0=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,0);
        z1=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,1);
        display("distance 0="+d(sumstats(z0),y))
        display("distance 1="+d(sumstats(z1),y))
          % Reject sample
        if d(sumstats(z0),y)<delta
            n0=n0+1;
        end
        if d(sumstats(z1),y)<delta
            n1=n1+1;
        end
    end
    display("n0="+n0+",n1="+n1)
    
    Lambda=n0/n1;
    yhat=Lambda<gamma;
   
end
