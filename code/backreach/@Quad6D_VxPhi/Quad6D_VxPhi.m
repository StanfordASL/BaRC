classdef Quad6D_VxPhi < DynSys
  
  properties
    % control bounds
    T1Max
    T1Min
    T2Max
    T2Min
    
    % "Real" parameters
    m % mass
    
    transDrag %translational drag
    
    % Ficticious parameter for decomposition
    wRange
  end
  
  methods
    function obj = Quad6D_VxPhi(x, T1Min, T1Max, T2Min, T2Max, wRange, ...
        m, transDrag)
      % Dynamics:
      %    \dot v_x  = -transDrag*v_x/m - T1*sin(\phi)/m - T2*sin(\phi)/m
      %    \dot \phi = \omega
      %
      % Inputs:
      %   T1Max, T1Min, T2Max, T2Min - limits on T1 and T2 (controls
      %   m - mass
      %   grav - gravity
      %   transDrag - translational Drag
      
      if nargin < 2
        T1Max = 36.7875/2;
      end
      
      if nargin < 3
        T1Min = 0;
      end
      
      if nargin < 4
        T2Max = 36.7875/2;
      end
      
      if nargin < 5
        T2Min = 0;
      end
      
      if nargin < 6
        wRange = [0 2*pi];
      end      

      if nargin < 7
        m = 1.25; %kg
      end      
      
      if nargin < 8
        transDrag = 0.25;
      end

      % Basic vehicle properties
      obj.nx = 2;
      obj.nu = 3;  
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.T1Max = T1Max;
      obj.T1Min = T1Min;
      obj.T2Max = T2Max;
      obj.T2Min = T2Min;
      obj.m = m;
      obj.transDrag = transDrag;
      obj.wRange = wRange;
    end
    
  end % end methods
end % end classdef
