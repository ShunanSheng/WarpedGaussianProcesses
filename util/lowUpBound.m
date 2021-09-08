function [lb,ub]=lowUpBound(distname,M)
    % Given the distribution and the size, output the lower bound lb ,
    % upper bound ub for its range
    
    switch distname
        case "Gamma"
            lb=zeros(M,1);
            ub=[];
        case "Beta"
            lb=zeros(M,1);
            ub=ones(M,1);
        otherwise
            lb=[];
            ub=[];
    end
end