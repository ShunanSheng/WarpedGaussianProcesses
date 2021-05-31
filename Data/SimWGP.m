function z=SimWGP(hyp,meanfunc,covfunc,warpfunc,x)
    f=SimGP(hyp,meanfunc,covfunc,x);
    z=warpfunc(hyp.dist,f);
end
