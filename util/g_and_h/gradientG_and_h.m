function dgh=gradientG_and_h(x, g, h, loc, sca, tol)
    
    % Calculate the gradient of g_and_h pdf at x
    if ~exist('loc', 'var') || isempty(loc)
        loc = 0;
    end

    if ~exist('sca', 'var') || isempty(sca)
        sca = 1;
    end

    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-8;
    end

    [z, ~] = g_and_h_inverse((x - loc) / sca, g, h, tol);
    v = grad_g_and_h(z, g, h, loc, sca);
    A = gradientNorm(z)./(v.^2);
    
    d2g_and_h = d2gh;
    B = -normpdf(z)./(v.^3).* d2g_and_h;
    
    dgh = A+B;
    
    function d2g_and_h = d2gh
        if g ~= 0
            d2g_and_h = sca.*(exp(g.*z).*exp(h.*z.^2./2).*(g+2.*h.*z)+...
                (exp(g.*z)-1)./g.*exp(h.*z.^2./2).*((h.*z).^2+h));
        else
            d2g_and_h = sca.*(exp(h.*z.^2./2).*h.*z+exp(h.*z.^2./2).*(2.*h.*z+h.^2.*z.^3));
        end

    end

end