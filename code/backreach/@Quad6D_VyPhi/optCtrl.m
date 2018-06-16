function uOpt = optCtrl(obj, ~, x, deriv, uMode)
% uOpt = optCtrl(obj, t, y, deriv, uMode, dims)

%% Input processing
if nargin < 5
  uMode = 'max';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

%% Optimal control
uOpt = cell(obj.nu, 1);

det = cell(obj.nu, 1);
det{1} = deriv{1} .* cos(x{2}) / obj.m;
det{2} = deriv{1} .* cos(x{2}) / obj.m;
det{3} = deriv{2};

uMin = [obj.T1Min; obj.T2Min; obj.wRange(1)];
uMax = [obj.T1Max; obj.T2Max; obj.wRange(2)];

if strcmp(uMode, 'max')
  for i = 1:obj.nu
    uOpt{i} = (det{i} >= 0)*uMax(i) + (det{i} < 0)*uMin(i);
  end
  
elseif strcmp(uMode, 'min')
  for i = 1:obj.nu
    uOpt{i} = (det{i} >= 0)*uMin(i) + (det{i} < 0)*uMax(i);
  end
end

end