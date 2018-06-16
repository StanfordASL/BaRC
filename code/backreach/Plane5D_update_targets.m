function [targetXT, targetYT, targetV, targetW] = Plane5D_update_targets(state, radii, oldTargetXT, oldTargetYT, oldTargetV, oldTargetW)
% Boris Ivanovic, 2018-05-06

global gXT gYT gV gW; % Gotta make this usable to all other functions.

XTdims = [1 3];
YTdims = [2 3];
Vdim = 4;
Wdim = 5;

newCylinderXT = shapeCylinderSquared(gXT, [], state(XTdims), radii(XTdims(1)));
newCylinderYT = shapeCylinderSquared(gYT, [], state(YTdims), radii(YTdims(1)));
newCylinderV = shapeCylinderSquared(gV, [], state(Vdim), radii(Vdim));
newCylinderW = shapeCylinderSquared(gW, [], state(Wdim), radii(Wdim));

targetXT = shapeUnion(newCylinderXT, oldTargetXT);
targetYT = shapeUnion(newCylinderYT, oldTargetYT);
targetV = shapeUnion(newCylinderV, oldTargetV);
targetW = shapeUnion(newCylinderW, oldTargetW);

end
