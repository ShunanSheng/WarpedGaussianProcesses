%% get points inside Singapore
xv = S.X;
yv = S.Y;
in = inpolygon(xSp,ySp, xv', yv');

%% plot grid points inside Singapore
figure()
plot(xv,yv) % polygon
axis equal
hold on
plot(xSp(in),ySp(in),'r.') % points inside the polygon
hold off


%% partition the training and test data
clc, close all
rng(120)
stns_loc = horzcat(lat, long);
ntrain = floor(nstns * 1);
index_train = randperm(nstns, ntrain);
index_test = setdiff(1:nstns, index_train);

stns_train = stns_loc(index_train, :);
stns_test = stns_loc(index_test, :);

data_train = Data_ave_week(:, index_train);
data_test = Data_ave_week(:, index_test);

%% plot training and test stations
figure()
geoscatter(stns_train(:,1), stns_train(:,2),'r','^')
hold on
geoscatter(stns_test(:,1), stns_test(:,2),'b','^')
legend(["Training set","Test set"])
geobasemap streets-light

%%
clc, close all, clear all
 % External contour, rectangle.
x1 = [0 0 6 6];
y1 = [0 3 3 0];

% First hole contour, square.
x2 = [1 2 2 1];
y2 = [1 1 2 2];


% Compute face and vertex matrices.

poly1 = polyshape(x1,y1);
poly2 = polyshape(x2,y2);
polyout = subtract(poly1,poly2);

[x, y] = boundary(polyout);
[f, v] = poly2fv(x,y);
% Display the patch.
patch('Faces',f,'Vertices',v,'FaceColor','r','EdgeColor','none');
axis off, axis equal


%%
theta = linspace(0, 2*pi, 100);
x1 = cos(theta) - 0.5;
y1 = -sin(theta);    % -sin(theta) to make a clockwise contour
x2 = x1 + 1;
y2 = y1;
[xa, ya] = polybool('union', x1, y1, x2, y2);
[xb, yb] = polybool('intersection', x1, y1, x2, y2);
[xc, yc] = polybool('xor', x1, y1, x2, y2);
[xd, yd] = polybool('subtraction', x1, y1, x2, y2);

patch(xd, yd, 1, 'FaceColor', 'r')
axis equal, axis off, hold on
plot(x1, y1, x2, y2, 'Color', 'k')
title('Subtraction')

%%

clc, close all, clear all
% S = shaperead('landareas', 'UseGeoCoords', true,...
%   'Selector','Singapore');
% % 

% S = shaperead('TM_WORLD_BORDERS_SIMPL-0.3.shp');
%
S = shaperead('SGP_adm0.shp');
lnlim = [min(S.X) max(S.X)];
ltlim = [min(S.Y) max(S.Y)];
x1 = [lnlim(1) lnlim(2) lnlim(2) lnlim(1)];
y1 = [ltlim(1) ltlim(1) ltlim(2) ltlim(2)];
x2 = S.X;
y2 = S.Y;
%%
poly1 = polyshape(x1,y1);
poly2 = polyshape(x2,y2);
polyout = subtract(poly1,poly2);

%% 
[x, y] = boundary(polyout);
[f, v] = poly2fv(x,y);

%%
clc, close all, clear all
file = 'SGP_adm0.shp';
S = shaperead(file);
[f, v] = maskPatch(S);

lnlim = [min(S.X) max(S.X)];
ltlim = [min(S.Y) max(S.Y)];
    

nx = 100;
ny = 20;
xq = linspace(lnlim(1), lnlim(2), nx);
yq = linspace(ltlim(1), ltlim(2), ny);
[xq,yq] = meshgrid(xq,yq);
X = reshape(xq,[],1);
Y = reshape(yq,[],1);
a = 1;
b = 9;
z = a + (b-a).*rand(ny,nx);

%%
% 
% figure()
% surf(xq,yq,z, "DisplayName","latent spatial field");
% shading INTERP
% view(2)
% colorbar

mapshow(xq, yq, z, 'DisplayType','surface', ...
                 'zdata', ones(size(xq))*0, ... % keep below gridlines
                 'cdata', z, ... 
                 'facecolor', 'flat');          % to show coarseness

%%
hold on
patch('Faces',f,'Vertices',v,'FaceColor','w','EdgeColor','none');
axis equal




%%
figure()
plot(poly1)
figure()
plot(poly2)


%%

[f, v] = poly2fv(xmask, ymask);


patch('Faces',f,'Vertices',v,'FaceColor','r','EdgeColor','none');
axis off, axis equal


%%


%%

ix = find(isnan(S.X),1);


figure()
plot(S.X,S.Y,'r')
hold on 
plot(S.X(1:ix-1),S.Y(1:ix-1),'b')
hold off
%%
lnlim = [min(S.X) max(S.X)];
ltlim = [min(S.Y) max(S.Y)];

% Some coarse fake data

nx = 100;
ny = 20;
xq = linspace(lnlim(1), lnlim(2), nx);
yq = linspace(ltlim(1), ltlim(2), ny);
[xq,yq] = meshgrid(xq,yq);
xq = reshape(xq,[],1);
yq = reshape(yq,[],1);
a = 1;
b = 9;
z = a + (b-a).*rand(ny,nx);

%%
figure()
scatter(xq,yq,'r','.')
hold on 
plot(S.X,S.Y,'b')

%%
mapshow(S.X,S.Y)


% find points inside the boundary
%%
xv = S.X(3626:7659);
yv = S.Y(3626:7659);
plot(xv,yv,'b')
%%

xv = S.X;
yv = S.Y;
in = inpolygon(xq,yq, xv', yv');


%%
figure

plot(xv,yv) % polygon
axis equal

hold on
plot(xq(in),yq(in),'r+') % points inside
% plot(xq(~in),yq(~in),'bo') % points outside
hold off


%%
xv = [0 3 3 0 0 NaN 1 1 2 2 1];
yv = [0 0 3 3 0 NaN 1 2 2 1 1];
xq = rand(1000,1)*3; yq = rand(1000,1)*3;
in = inpolygon(xq,yq,xv,yv);
plot(xv,yv,xq(in),yq(in),'.r',xq(~in),yq(~in),'.b')

%%
L = linspace(0,2*pi,6);
x1 = cos(L)';
y1 = sin(L)';

x2 = cos(L)' + 2;
y2 = sin(L)' + 2;

xv = [x1;NaN;x2];
yv = [y1;NaN;y2];

rng default
xq = randn(250,1);
yq = randn(250,1);

[in,on] = inpolygon(xq,yq,xv,yv);


figure

plot(xv,yv) % polygon
axis equal

hold on
plot(xq(in),yq(in),'r+') % points inside
plot(xq(~in),yq(~in),'bo') % points outside
hold off


%%
clc, close all, clear all

%%
S = shaperead('landareas', 'UseGeoCoords', true,...
  'Selector',{@(name) strcmp(name,'Australia'), 'Name'});



ltlim = [min(S.Lat) max(S.Lat)] + [-1 1];
lnlim = [min(S.Lon) max(S.Lon)] + [-1 1];
% Some coarse fake data
nx = 20;
ny = 20;
x = linspace(lnlim(1), lnlim(2), nx);
y = linspace(ltlim(1), ltlim(2), ny);
[x,y] = meshgrid(x,y);
a = 1;
b = 9;
z = a + (b-a).*rand(ny,nx);
% Plot data as surface
figure('color','w');
worldmap('Australia');
geoshow(y, x, z, 'DisplayType','surface', ...
                 'zdata', ones(size(x))*-1, ... % keep below gridlines
                 'cdata', z, ... 
                 'facecolor', 'flat');          % to show coarseness
geoshow(S, 'edgecolor', 'k', 'facecolor', 'none');
% Create mask overlay
%%
ltbox = ltlim([1 2 2 1 1]);
lnbox = lnlim([1 1 2 2 1]);
[ltbox, lnbox] = interpm(ltbox, lnbox, 1);

%%
[xbox, ybox] = mfwdtran(ltbox, lnbox); % Box around data
[xaus, yaus] = mfwdtran(S.Lat, S.Lon); % Australia polygon
[xmask, ymask] = polybool('-', xbox, ybox, xaus, yaus); % box w/ Aus. hole

%%
[f,v] = poly2fv(xmask, ymask);
hmask = patch('faces', f, 'vertices', v, ...
    'facecolor', 'w', 'edgecolor', 'none');