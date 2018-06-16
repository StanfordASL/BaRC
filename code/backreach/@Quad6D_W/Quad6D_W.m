classdef Quad6D_W < DynSys
  
  properties
    % control bounds
    T1Max
    T1Min
    T2Max
    T2Min
    
    I % Moment of inertia
    l % length of quadrotor
    rotDrag %translational drag
  end
  
  methods
    function obj = Quad6D_W(x, T1Min, T1Max, T2Min, T2Max, I, l, rotDrag)
      % Dynamics:
      %    \dot \omega  = rotDrag*\omega/I - l*u1/I + l*u2/I

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
        I = 0.03; 
      end
      
      if nargin < 7
        l = 0.5;
      end
      
      if nargin < 8
        rotDrag = 0.02255;
      end      
      
      % Basic vehicle properties
      obj.nx = 1;
      obj.nu = 2;  
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.T1Max = T1Max;
      obj.T1Min = T1Min;
      obj.T2Max = T2Max;
      obj.T2Min = T2Min;
      obj.I = I;
      obj.l = l;
      obj.rotDrag = rotDrag;

    end
    
  end % end methods
end % end classdef
