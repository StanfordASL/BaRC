function dx = dynamics(obj, ~, x, u, ~)
      % Dynamics:
      %    \dot \omega  = rotDrag*\omega/I - l*u1/I + l*u2/I
      
dx = cell(obj.nx,1);

returnVector = false;
if ~iscell(x)
  returnVector = true;
  x = num2cell(x);
  u = num2cell(u);
end

for dim = 1:obj.nx
  dx{dim} = dynamics_cell_helper(obj, x, u, dim);
end

if returnVector
  dx = cell2mat(dx);
end
end

function dx = dynamics_cell_helper(obj, x, u, dim)

switch dim
  case 1
    dx = -obj.rotDrag*x{1}/obj.I - obj.l*u{1}/obj.I + obj.l*u{2}/obj.I;

  otherwise
    error('Only dimension 1 is defined for dynamics of Quad6D_W!')
end
end