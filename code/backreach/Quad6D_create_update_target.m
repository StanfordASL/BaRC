function [targetX, targetY, targetW, targetVxPhi, targetVyPhi] = ...
  Quad6D_create_update_target(gMin, gMax, gN, goalState, goalRadii)
% Boris Ivanovic, 2018-06-11

global gX gY gW gVxPhi gVyPhi;

%% Target and obstacles
Xdim = 1;
VxPhidims = [2 5];
Ydim = 3;
VyPhidims = [4 5];
Wdim = 6;

alreadyMade = sum(size(gX)) > 0;
if ~alreadyMade
    % Create grid structures for computation
    gX = createGrid(gMin(Xdim), gMax(Xdim), gN(Xdim));
    gY = createGrid(gMin(Ydim), gMax(Ydim), gN(Ydim));
    gW = createGrid(gMin(Wdim), gMax(Wdim), gN(Wdim));
    gVxPhi = createGrid(gMin(VxPhidims), gMax(VxPhidims), gN(VxPhidims), 2);
    gVyPhi = createGrid(gMin(VyPhidims), gMax(VyPhidims), gN(VyPhidims), 2);
end

%% Initial target set
targetX = shapeCylinder(gX, [], goalState(Xdim), goalRadii(Xdim));
targetY = shapeCylinder(gY, [], goalState(Ydim), goalRadii(Ydim));
targetW = shapeCylinder(gW, [], goalState(Wdim), goalRadii(Wdim));
targetVxPhi = shapeCylinder(gVxPhi, [], goalState(VxPhidims), goalRadii(VxPhidims(2)));
targetVyPhi = shapeCylinder(gVyPhi, [], goalState(VyPhidims), goalRadii(VyPhidims(2)));

end
