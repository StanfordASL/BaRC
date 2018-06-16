function [targetXT, targetYT, targetV, targetW] = ...
  Plane5D_create_init_target(gMin, gMax, gN, goalState, goalRadii)
% Boris Ivanovic, 2018-05-06

global gXT gYT gV gW;

%% Target and obstacles
% 5D grid limits (x, y, \theta, v, \omega)
if nargin < 3
  gMin = [-5; -5; -pi; 0; -1];
  gMax = [5; 5; pi; 3; 1];
  gN = 101*ones(5,1);
end

XTdims = [1 3];
YTdims = [2 3];
Vdim = 4;
Wdim = 5;

alreadyMade = sum(size(gXT)) > 0;
if ~alreadyMade
    % Create grid structures for computation
    gXT = createGrid(gMin(XTdims), gMax(XTdims), gN(XTdims), 2);
    gYT = createGrid(gMin(YTdims), gMax(YTdims), gN(YTdims), 2);
    gV = createGrid(gMin(Vdim), gMax(Vdim), gN(Vdim), 2);
    gW = createGrid(gMin(Wdim), gMax(Wdim), gN(Wdim), 2);
end

%% Initial target set
targetXT = shapeCylinderSquared(gXT, [], goalState(XTdims), goalRadii(XTdims(1)));
targetYT = shapeCylinderSquared(gYT, [], goalState(YTdims), goalRadii(YTdims(2)));
targetV = shapeCylinderSquared(gV, [], goalState(Vdim), goalRadii(Vdim));
targetW = shapeCylinderSquared(gW, [], goalState(Wdim), goalRadii(Wdim));

end
