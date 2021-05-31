function f=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,yx)
    if yx==0
%         display("yx=="+yx)
        f=SimWGP(hyp0,meanfunc0,covfunc0,warpfunc,t);
    else
%         display("yx=="+yx)
        f=SimWGP(hyp1,meanfunc1,covfunc1,warpfunc,t);
end