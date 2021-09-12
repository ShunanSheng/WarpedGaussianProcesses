function plotVector(x,Xtitle,Xlegend)
    if ~exist('Xtitle', 'var')||isempty(Xtitle)
        Xtitle = "Plot of x";
    end
    if ~exist('Xlegend', 'var')||isempty(Xlegend)
        Xlegend="x";
    end
    n=length(x);
    plot(1:1:n,x);
    title(Xtitle);
    legend(Xlegend);

end

