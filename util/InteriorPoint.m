function [x,fval]=InteriorPoint(f,x0,lb,ub)
    [x,fval] = fmincon(f,x0,[],[],[],[],lb,ub);
end
