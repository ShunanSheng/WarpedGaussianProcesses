function [x,fval]=InteriorPoint(f,x0,lb,ub)
    % Barrier method 
%     options = optimoptions('fmincon','Display','iter');
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
    [x,fval] = fmincon(f,x0,[],[],[],[],lb,ub,[],options);
end
