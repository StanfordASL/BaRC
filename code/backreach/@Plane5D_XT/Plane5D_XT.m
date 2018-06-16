classdef Plane5D_XT < DynSys
  properties
    % Speed bounds
    vRange
    
    % Turn rate bounds
    wMax
  end
  
  methods
    function obj = Plane5D_XT(~, vRange, wMax)
      % obj = Plane5D_XT(~, vRange, wMax)
      %
      %     Dynamics of Plane5D_XT
      %         \dot{x}_1 = d_1 * cos(x_2) (x position)
      %         \dot{x}_2 = u_1            (heading)
      %             u_1 -- turn rate
      %             d_1 -- speed
      %
      %     For reference: Dynamics of the Plane5D
      %         \dot{x}_1 = x_4 * cos(x_3) (x position)
      %         \dot{x}_2 = x_4 * sin(x_3) (y position)
      %         \dot{x}_3 = x_5                  (heading)
      %         \dot{x}_4 = u_1            (linear speed)
      %         \dot{x}_5 = u_2           (turn rate)
      %           aRange(1) <= u_1 <= aRange(2)
      %           -alphaMax <= u_2 <= alphaMax
      
      obj.vRange = vRange;
      obj.wMax = wMax;
      
      obj.nx = 2;
      obj.nu = 1;
      obj.nd = 1;
    end
  end % end methods
end % end classdef
