function dgh=grad_g_and_h(z, g, h, loc, sca)
    % the derivative of the g_and_h transformation
    if ~exist('loc', 'var') || isempty(loc)
        loc = 0;
    end

    if ~exist('sca', 'var') || isempty(sca)
        sca = 1;
    end
    
    if g~=0
        dgh=sca.*(exp(g.*z).*exp(h.*z.^2./2)+(exp(g.*z)-1)./g.*exp(h.*z.^2./2).*h.*z);
    else
        dgh=sca.*(exp(h.*z.^2./2)+z.*exp(h.*z.^2./2).*h.*z);
    end

end