function [x,fval]=InteriorPoint(f,x0,lb,ub,options)
    %%% SQP method
    [x,fval] = fmincon(f,x0,[],[],[],[],lb,ub,[],options);
end
