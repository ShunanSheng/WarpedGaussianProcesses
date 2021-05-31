function zI=SimIntData(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,K,snI,yx)
    zI=zeros(K,1);T=hyp0.t;n0=100; % find a dense grid on [0,T]
    t=linspace(0,T,K*n0)';
    display("n="+K*n0)
    z=SimTemporal(hyp0,hyp1,meanfunc0,covfunc0,meanfunc1,covfunc1,warpfunc,t,yx);
    size(z,1)
    for i=1:K
        zI(i)=T/K/n0*sum(z((i-1)*n0+1:i*n0)); % Take riemann-stiejtes sum
    end
    zI=zI+snI*randn(K,1);
end