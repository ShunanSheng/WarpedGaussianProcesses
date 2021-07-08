function [x,fval]=InteriorPoint(f,x0,lb,ub)
    % Barrier method 
    options = optimoptions('fmincon','Display','off');
    [x,fval] = fmincon(f,x0,[],[],[],[],lb,ub,[],options);
end
