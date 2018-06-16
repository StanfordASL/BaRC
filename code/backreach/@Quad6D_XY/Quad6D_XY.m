classdef Quad6D_XY < DynSys
  
  properties
    % control bounds
    vRange
  end
  
  methods
    function obj = Quad6D_XY(x, vRange)
      % Dynamics:
      %    \dot x or \dot y = vxRange or vyRange
      %
      % Inputs:
      %   vRange - estimated velocity range
      
      % Basic vehicle properties
      obj.nx = 1;
      obj.nu = 1;  
      
      obj.x = x;
      obj.xhist = obj.x;
      obj.vRange = vRange;
    end
    
  end % end methods
end % end classdef
