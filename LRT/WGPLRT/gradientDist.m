function df=gradientDist(pd,v)
    % Calculate the gradient of pdf of the given distribution 
    % Input:
    % pd: the probabilty distribution
    % v : the point to take gradient
    % Output: 
    % df: the gradient of pd at v (n x 1)
%     if strcmp(pd.DistributionName,'g and h')
%         df=grad_g_and_h(v, g, h, loc, sca)
%     end
    h=0.00001;
    V=[v,v+h];
    y=pdf(pd,V);
    df=diff(y,1,2)/h;
end