function dx = dynamics(obj, ~, ~, u, ~)

dx = cell(obj.nx,1);

returnVector = false;
if ~iscell(u)
  returnVector = true;
  u = num2cell(u);
end

for dim = 1:obj.nx
  dx{dim} = dynamics_cell_helper(u, dim);
end

if returnVector
  dx = cell2mat(dx);
end
end

function dx = dynamics_cell_helper(u, dim)

switch dim
  case 1
    dx = u{1};

  otherwise
    error('Only dimension 1 is defined for dynamics of Quad6D_W!')
end
end