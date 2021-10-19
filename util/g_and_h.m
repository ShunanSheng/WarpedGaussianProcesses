function x=g_and_h(z, g, h, loc, sca)
    % Perform g and h transformation given z ~ N(0,1)
    % Inputs:
    %       z: follows N(0,1)
    %       g: the g parameter
    %       h: the h parameter
    %       loc: the location parameter, default is 0
    %       sca: the scale parameter, default is 1
    % Outputs:
    %       x: the transformed variable

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