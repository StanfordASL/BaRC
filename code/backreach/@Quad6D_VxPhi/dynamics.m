function dx = dynamics(obj, ~, x, u, ~)
      % Dynamics:
      %    \dot v_x  = -transDrag*v_x/m - T1*sin(\phi)/m - T2*sin(\phi)/m
      %    \dot \phi = \omega
      
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
    dx = (-obj.transDrag * x{1} / obj.m) + ...
          (-sin(x{2}) .* u{1} / obj.m) + ...
          (-sin(x{2}) .* u{2} / obj.m);
  case 2
    dx = u{3};

  otherwise
    error('Only dimension 1-2 are defined for dynamics of Quad6D_VxPhi!')
end
end