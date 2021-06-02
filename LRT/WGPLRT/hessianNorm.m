function d2f=hessianNorm(v)
    d2f=normpdf(v).*(v.^2-1);
end