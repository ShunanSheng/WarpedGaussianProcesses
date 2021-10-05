function x=g_and_h(z, g, h, loc, sca)
    % Perform g and h transformation 
    if ~exist('loc', 'var') || isempty(loc)
        loc = 0;
    end

    if ~exist('sca', 'var') || isempty(sca)
        sca = 1;
    end
    
    if g~=0
        x=loc+sca.*(exp(g.*z)-1)./g.*exp(h.*z.^2./2);
    else
        x=loc+sca.*z.*exp(h.*z.^2./2);
    end


end