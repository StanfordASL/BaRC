function uOpt = optCtrl(obj, ~, ~, deriv, uMode)
% uOpt = optCtrl(obj, t, y, deriv, uMode)
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

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

uOpt = cell(obj.nu, 1);

%% Optimal control
if strcmp(uMode, 'max')
  uOpt{1} = (deriv{2}>=0)*obj.wMax + (deriv{2}<0)*(-obj.wMax);

elseif strcmp(uMode, 'min')
  uOpt{1} = (deriv{2}>=0)*(-obj.wMax) + (deriv{2}<0)*obj.wMax;
  
else
  error('Unknown uMode!')
end

end