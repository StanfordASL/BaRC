function data = shapeCylinderSquared(grid, ignoreDims, center, radius)
% shapeCylinderSquared: implicit surface function for a cylinder.
%
%   data = shapeCylinderSquared(grid, ignoreDims, center, radius)
%
% Creates an implicit surface function (actually signed distance) for a
%   coordinate axis aligned cylinder whose axis runs parallel to the
%   coordinate dimensions specified in ignoreDims.
%
% Can be used to create:
%   Intervals, circles and spheres (if ignoreDims is empty).
%   Slabs (if ignoreDims contains all dimensions except one).
%
% parameters:
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).
%
%   ignoreDims: Vector specifying indices of coordinate axes with which the
%   cylinder is aligned.  Defaults to the empty vector (eg: the cylinder is
%   actually a sphere).
%
%   center: Vector specifying a point at the center of the cylinder.
%   Entries in the ignored dimensions are ignored.  May be a scalar, in
%   which case the scalar is multiplied by a vector of ones of the
%   appropriate length.  Defaults to 0 (eg centered at the origin).
%
%   radius: Scalar specifying the radius of the cylinder.  Defaults to 1.
%
% Output Parameters:
%
%   data: Output data array (of size grid.size) containing the implicit
%   surface function.

%---------------------------------------------------------------------------
% Default parameter values.
if(nargin < 2)
  ignoreDims = [];
end

if(nargin < 3)
  center = zeros(grid.dim, 1);
elseif(numel(center) == 1)
  center = center * ones(grid.dim, 1);
end

if(nargin < 4)
  radius = 1;
end

%---------------------------------------------------------------------------
% Signed distance function calculation.
data = zeros(grid.shape);
for i = 1 : grid.dim
  if(all(i ~= ignoreDims))
    data = data + (grid.xs{i} - center(i)).^2;
  end
end
data = data - radius.^2;

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
