function F = rectfeature(II,RECTS,FR,ORDER,VERT)
% function F = rectfeature(II,RECTS,FR,ORDER,VERT)
% Compute matrix of features from a stack of integral images
%
% II [MxNxK] - stack of K integral images, each MxN
% RECTS [Kx(4*NR)] - NR rectangles per image: integer coordinates
% FR [1x4] - fraction of each rect to use for feature: [fx,fy,fw,fh]
%   0 <= FR(1:4) <= 1
% ORDER [1x1]: 1 <= ORDER <= 4 means the number of sub-rectangles
% VERT [1x1] = 1 if vertical orientation, 0 if horizontal
%
% F [KxNR] - computed values of this feature for each training rectangle
%
% Mark Hasegawa-Johnson
% 4/3/2016
% Modified 4/13/2016 to have more consistent notation

% Number of rectangles per image
NR = size(RECTS,2)/4;
% Number of images in which to compute rectangles
K = min(size(RECTS,1),size(II,3));
% One feature for each rectangle, in each image
F = zeros(K,NR);

z = zeros(1,NR);
o = ones(1,NR);

for k=1:K,
    if mod(k, 10) == 0
        disp(k);
    end
   % Rectangle for this feature in this image:
   % [xmin,ymin]=BASE(xmin,ymin)+FR(1:2)*BASE(width,height)
   % [width,height]=FR(3:4)*BASE(width,height)
   b = RECTS(k,:);
   BASE = reshape(b, 4, NR);
   rect = [BASE(1:2,:);z;z] + round(FR' .* [BASE(3:4,:);BASE(3:4,:)]);

   % Default coordinates: one rectangle
   xcoords = rect(1,:)+[z;rect(3,:)];
   ycoords = rect(2,:)+[z;rect(4,:)];
   xcofs = [-1,1];
   ycofs = [-1,1];

   % Coordinates from which to sample II
   if (((ORDER==2) && (VERT==1)) || (ORDER==4)),
     xcoords = rect(1,:)+round([0,0.5,1]'*rect(3,:));
     xcofs = [1,-2,1];
   end
   if (((ORDER==2) && (VERT==0)) || (ORDER==4)),
     ycoords = size(II,1),rect(2,:)+round([0,0.5,1]'*rect(4,:));
     ycofs = [1,-2,1];
   end
   if ((ORDER==3) && (VERT==1)),
     xcoords = rect(1,:)+round([0,0.33,0.67,1]'*rect(3,:));
     xcofs = [-1,2,-2,1];
   end
   if ((ORDER==3) && (VERT==0)),
     ycoords = rect(2,:)+round([0,0.3,0.67,1]*rect(4,:));
     ycofs = [-1,2,-2,1];
   end

   % Sample the integral image
%    F(k,nr) = F(k,nr) + ycofs*II(ycoords,xcoords,k)*xcofs';
    cofs = ycofs' * xcofs;
    cofs = cofs(:);
    ivals = zeros(size(cofs,1), NR);
    
    ycoords(ycoords > size(II,1)) = size(II,1);
    xcoords(xcoords > size(II,2)) = size(II,2);

    for i = 1:NR
        iv = II(ycoords(:,i),xcoords(:,i),k);
        ivals(:,i) = iv(:);
    end
    F(k,:) = sum(cofs .* ivals);
end
