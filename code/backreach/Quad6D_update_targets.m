function [targetX, targetY, targetW, targetVxPhi, targetVyPhi] = ...
  Quad6D_update_targets(state, radii, oldTargetX, oldTargetY, oldTargetW, oldTargetVxPhi, oldTargetVyPhi)
% Boris Ivanovic, 2018-06-11

global gX gY gW gVxPhi gVyPhi;

%% Target and obstacles
Xdim = 1;
VxPhidims = [2 5];
Ydim = 3;
VyPhidims = [4 5];
Wdim = 6;

newCylinderX = shapeCylinder(gX, [], state(Xdim), radii(Xdim));
newCylinderY = shapeCylinder(gY, [], state(Ydim), radii(Ydim));
newCylinderW = shapeCylinder(gW, [], state(Wdim), radii(Wdim));
newCylinderVxPhi = shapeCylinder(gVxPhi, [], state(VxPhidims), radii(VxPhidims(2)));
newCylinderVyPhi = shapeCylinder(gVyPhi, [], state(VyPhidims), radii(VyPhidims(2)));

targetX = shapeUnion(newCylinderX, oldTargetX);
targetY = shapeUnion(newCylinderY, oldTargetY);
targetW = shapeUnion(newCylinderW, oldTargetW);
targetVxPhi = shapeUnion(newCylinderVxPhi, oldTargetVxPhi);
targetVyPhi = shapeUnion(newCylinderVyPhi, oldTargetVyPhi);

end
