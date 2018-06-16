function dOpt = optDstb(obj, ~, x, deriv, dMode)
% uOpt = optCtrl(obj, t, y, deriv, uMode)
%     Dynamics of the Plane5D
%         \dot{x}_1 = x_4 * cos(x_3) + d_1 (x position)
%         \dot{x}_2 = x_4 * sin(x_3) + d_2 (y position)
%         \dot{x}_3 = x_5                  (heading)
%         \dot{x}_4 = u_1 + d_3            (linear speed)
%         \dot{x}_5 = u_2 + d_4            (turn rate)

%% Input processing
if nargin < 5
  dMode = 'max';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

dOpt = cell(obj.nd, 1);

%% Optimal disturbance
det = deriv{1}.* cos(x{2});

if strcmp(dMode, 'max')
  dOpt{1} = (det>=0)*obj.vRange(2) + (det<0)*obj.vRange(1);

elseif strcmp(dMode, 'min')
  dOpt{1} = (det>=0)*obj.vRange(1) + (det<0)*obj.vRange(2);
  
else
  error('Unknown dMode!')
end

end