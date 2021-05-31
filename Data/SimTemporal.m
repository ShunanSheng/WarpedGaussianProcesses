function f=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,yx)
    if yx==0
        f=SimWGP(hyp0,meanfunc0,covfunc0,warpfunc,t);
    else
        f=SimWGP(hyp1,meanfunc1,covfunc1,warpfunc,t);
end