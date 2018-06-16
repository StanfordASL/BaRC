function dx = dynamics(obj, ~, x, u, d)
% dx = dynamics(obj, ~, x, u, d)
%     Dynamics of Plane5D_YT
%         \dot{x}_1 = d_1 * sin(x_2) (x position)
%         \dot{x}_2 = u_1            (heading)
%             u_1 -- turn rate
%             d_1 -- speed
%
%     For reference: Dynamics of the Plane5D
%         \dot{x}_1 = x_4 * cos(x_3) + d_1 (x position)
%         \dot{x}_2 = x_4 * sin(x_3) + d_2 (y position)
%         \dot{x}_3 = x_5                  (heading)
%         \dot{x}_4 = u_1 + d_3            (linear speed)
%         \dot{x}_5 = u_2 + d_4            (turn rate)

dx = cell(obj.nx, 1);

returnVector = false;
if ~iscell(x)
  returnVector = true;
  x = num2cell(x);
  u = num2cell(u);
end

for i = 1:obj.nx
  dx{i} = dynamics_i(x, u, d, i);
end

if returnVector
  dx = cell2mat(dx);
end
end

function dx = dynamics_i(x, u, d, dim)

switch dim
  case 1
    dx = d{1} .* sin(x{2});
  case 2
    dx = u{1};
  otherwise
    error('Only dimension 1-2 are defined for dynamics of Plane5D_XT!')    
end
end