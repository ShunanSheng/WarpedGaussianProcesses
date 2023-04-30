% pd = makedist("Beta", "a", 2, "b", 5);
% pd = makedist("Gamma", "a", 2, "b", 5);
% pd = makedist("Exponential", 2);
% pd = makedist("tLocationScale", "mu", 1, "sigma", 1, "nu",5);
pd = makedist("g_and_h","g",0.1,"h",0.4,'loc',1,'sca',1);
warpfunc = @(pd,p) invCdf(pd,p);
warpinv = @(pd,p) invCdfWarp(pd,p);
W = @(x) warpfunc(pd,x);
G = @(v) warpinv(pd,v);
N = 10000;
x = sort(normrnd(0, 1, [N, 1]), 'ascend');
v = W(x);
dG = gradientG(pd,G,v);

plot(x, log(dG))

% histogram(v)

%%
pd = makedist('Normal');
t = truncate(pd,-2,Inf);
x = linspace(-3,10,1000);
figure
plot(x,pdf(pd,x))
hold on
plot(x,pdf(t,x),'LineStyle','--')
legend('Normal','Truncated')
hold off

%%
histogram(random(t, 100000, 1))