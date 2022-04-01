function [f, v] = maskPatch(S)
    % given the shape file, export parameters for patch 
    
    lnlim = [min(S.X) max(S.X)];
    ltlim = [min(S.Y) max(S.Y)];
    
    % define vertices for polygons
    x1 = [lnlim(1) lnlim(2) lnlim(2) lnlim(1)];
    y1 = [ltlim(1) ltlim(1) ltlim(2) ltlim(2)];
    x2 = S.X;
    y2 = S.Y;
    
    % subtract the shape of the region from the map
    poly1 = polyshape(x1,y1);
    poly2 = polyshape(x2,y2);
    polyout = subtract(poly1,poly2);

    % get output and convert contours to face 
    [x, y] = boundary(polyout);
    [f, v] = poly2fv(x,y);

end 