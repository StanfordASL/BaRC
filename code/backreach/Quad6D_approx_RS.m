function [dataX, dataY, dataW, dataVxPhi, dataVyPhi] = ...
  Quad6D_approx_RS(gMin, gMax, gN, T1Min, T1Max, T2Min, T2Max, wRange, ...
  vxRange, vyRange, targetX, targetY, targetW, targetVxPhi, targetVyPhi, tMax)
%
% Computes the approximate backward reachable set for the 6D quadrotor
% model by decomposing the system.
%
% This function requires the helperOC and toolboxls libraries, found at
%     https://github.com/HJReachability/helperOC.git
%     https://bitbucket.org/ian_mitchell/toolboxls
%
% Inputs:
%   - (gMin, gMax, gN):
%         grid parameters
%   - (T1Min, T1Max, T2Min, T2Max, wRange, vxRange, vyRange):
%         system/subsystem parameters;
%         wRange, vxRange, vyRange are needed because of decomposition, which
%         produces the subsystems x, y, \omega, (v_x, \phi), (v_y, \phi)
%   - tMax:
%         Time horizon of reachable set
%   - (targetX, targetY, targetW, targetVxPhi, targetVyPhi):
%         target sets for subsystems x, y, \omega, (v_x, \phi), (v_y, \phi)
%
% Outputs:
%   - (dataX, dataY, dataW, dataVxPhi, dataVyPhi)
%         value functions for subsystems x, y, \omega, (v_x, \phi), (v_y, \phi)
%         a state is in over-approximation of reachable set if all value 
%           functions evaluate to negative
%
% Explanation of decomposition:
%     0. 6D state: (x, v_x, y, v_y, \phi, \omega)
%
%     1. 4D subsystems: (x, v_x, \phi, \omega), (y, v_y, \phi, \omega)
%           See Chen et al., "Decomposition of Reachable Sets and Tubes for
%           a Class of Nonlinear Systems," TAC 2018
%
%     2a. Subsystem (x, v_x, \phi, \omega) becomes three subsystems:
%             x, \omega, (v_x, \phi)
%     2b. Subsystem (y, v_y, \phi, \omega) becomes three subsystems:
%             y, \omega, (v_y, \phi)
%     2c. Subsystem \omega can be reused
%
%     3. Reachable sets in each subsystem is represented by the zero
%        sublevel set of a value function.
%
%     4a. The 6D reachable set is over-approximated by the maximum of all
%         the value functions.
%     4b. Equivalently, the 6D reachable set over-approximated by the
%         intersection of backprojection of subsystem reachable sets
%     4c. Equivalently, a point (x, v_x, y, v_y, \phi, \omega) is in the
%         over-approximation if the state variables evaluate to a negative
%         value on all subsystem value functions
%
% Mo Chen, 2018-05-30

global gX gY gW gVxPhi gVyPhi; % Gotta make this usable from other functions.

%% Target and obstacles
% 6D grid limits (x, y, \theta, v, \omega)
if nargin < 3
  gMin = [-5;   -3;  -5; -10;    0;    0];
  gMax = [5;    18;   5;  12; 2*pi; 2*pi];
  gN =   [251;  81; 251;  81;  101;   51];
end

% Subsystem dimensions
Xdim = 1;
VxPhidims = [2 5];
Ydim = 3;
VyPhidims = [4 5];
Wdim = 6;

% Create grid structures for computation
% gX = createGrid(gMin(Xdim), gMax(Xdim), gN(Xdim));
% gY = createGrid(gMin(Ydim), gMax(Ydim), gN(Ydim));
% gW = createGrid(gMin(Wdim), gMax(Wdim), gN(Wdim));
% gVxPhi = createGrid(gMin(VxPhidims), gMax(VxPhidims), gN(VxPhidims), 2);
% gVyPhi = createGrid(gMin(VyPhidims), gMax(VyPhidims), gN(VyPhidims), 2);

%% Quadrotor parameters
% Constants
g = 9.81;
m = 1.25;
if nargin < 4
  % thrust range
  T1Min = 0;
  T1Max = 1.25*m*g;
  T2Min = 0;
  T2Max = 1.25*m*g;
  
  % fictitious parameters
  wRange = [0 2*pi];
  vxRange = [-5 15];
  vyRange = [-10 10];
end

%% Time parameters
if nargin < 11
  tMax = 0.5;
end

 % Time horizon and intermediate results
tau = 0:0.05:tMax;

%% Dynamical subsystems
q_X = Quad6D_XY([], vxRange);
q_Y = Quad6D_XY([], vyRange);
q_W = Quad6D_W([], T1Min, T1Max, T2Min, T2Max);
q_VxPhi = Quad6D_VxPhi([], T1Min, T1Max, T2Min, T2Max, wRange); 
q_VyPhi = Quad6D_VyPhi([], T1Min, T1Max, T2Min, T2Max, wRange); 

%% Default target set
if nargin < 12
  target = [0; 3; 0; 0; pi; pi];
  targetX = (gX.xs{1} - target(1)).^2/0.1^2 - 1;
  targetY = (gY.xs{1} - target(3)).^2/0.1^2 - 1;
  targetW = (gW.xs{1} - target(6)).^2/(30*pi/180)^2 - 1;
  targetVxPhi = max((gVxPhi.xs{1} - target(2)).^2/0.5^2 - 1, ...
                (gVxPhi.xs{2} - target(5)).^2/(20*pi/180)^2 - 1);
  targetVyPhi = max((gVyPhi.xs{1} - target(4)).^2/0.5^2 - 1, ...
                (gVyPhi.xs{2} - target(5)).^2/(20*pi/180)^2 - 1);
end

%% Compute reachable set
% Solver parameters
uMode = 'min';
vis = false;
quiet = true;
keepLast = false;

sDX.dynSys = q_X;
sDX.grid = gX;
sDX.uMode = uMode;
eAX.visualize = vis;
eAX.quiet = quiet;
eAX.keepLast = keepLast;

sDY.dynSys = q_Y;
sDY.grid = gY;
sDY.uMode = uMode;
eAY.visualize = vis;
eAY.quiet = quiet;
eAY.keepLast = keepLast;

sDW.dynSys = q_W;
sDW.grid = gW;
sDW.uMode = uMode;
eAW.visualize = vis;
eAW.quiet = quiet;
eAW.keepLast = keepLast;

sDVxPhi.dynSys = q_VxPhi;
sDVxPhi.grid = gVxPhi;
sDVxPhi.uMode = uMode;
eAVxPhi.visualize = vis;
eAVxPhi.quiet = quiet;
eAVxPhi.keepLast = keepLast;

sDVyPhi.dynSys = q_VyPhi;
sDVyPhi.grid = gVyPhi;
sDVyPhi.uMode = uMode;
eAVyPhi.visualize = vis;
eAVyPhi.quiet = quiet;
eAVyPhi.keepLast = keepLast;

% Call solver
dataX = HJIPDE_solve(targetX, tau, sDX, 'none', eAX);
dataY = HJIPDE_solve(targetY, tau, sDY, 'none', eAY);
dataW = HJIPDE_solve(targetW, tau, sDW, 'none', eAW);
dataVxPhi = HJIPDE_solve(targetVxPhi, tau, sDVxPhi, 'none', eAVxPhi);
dataVyPhi = HJIPDE_solve(targetVyPhi, tau, sDVyPhi, 'none', eAVyPhi);

end
