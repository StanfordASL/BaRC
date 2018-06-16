function [dataXT, dataYT, dataV, dataW] = ...
  Plane5D_approx_RS(gMin, gMax, gN, aMax, alphaMax, wMax, vRange, tMax, ...
  targetXT, targetYT, targetV, targetW)
% [dataXT, dataYT, dataV, dataW] = ...
%   Plane5D_approx_RS(gMin, gMax, gN, aMax, alphaMax, wMax, vRange, ...
%   targetXT, targetYT, targetV, targetW)
%
% Computes the approximate backward reachable set for the 5D plane (or car)
% model by decomposing the system.
%
% This function requires the helperOC and toolboxls libraries, found at
%     https://github.com/HJReachability/helperOC.git
%     https://bitbucket.org/ian_mitchell/toolboxls
%
% Inputs:
%   - (gMin, gMax, gN):
%         grid parameters
%   - (aMax, alphaMax, wMax, vRange):
%         system/subsystem parameters;
%         wMax and vRange are needed because of decomposition, which
%         produces the subsystems (x, \theta), (y, \theta), v, \omega
%   - tMax:
%         Time horizon of reachable set
%   - (targetXT, targetYT, targetV, targetW):
%         target sets for subsystems (x, \theta), (y, \theta), v, \omega
%
% Outputs:
%   - (dataXT, dataYT, dataV, dataW)
%         value functions for subsystems (x, \theta), (y, \theta), v, \omega
%         a state is in over-approximation of reachable set if all value 
%           functions evaluate to negative
%
% Explanation of decomposition:
%     0. 5D state: (x, y, \theta, v, \omega)
%
%     1. 4D subsystems: (x, \theta, v, \omega), (y, \theta, v, \omega)
%           See Chen et al., "Decomposition of Reachable Sets and Tubes for
%           a Class of Nonlinear Systems," TAC 2018
%
%     2a. Subsystem (x, \theta, v, \omega) becomes three subsystems:
%             (x, \theta), v, \omega
%     2b. Subsystem (y, \theta, v, \omega) becomes three subsystems:
%             (y, \theta), v, \omega
%     2c. Subsystems v, \omega can be reused
%
%     3. Reachable sets in each subsystem is represented by the zero
%        sublevel set of a value function.
%
%     4a. The 5D reachable set is over-approximated by the maximum of all
%         the value functions.
%     4b. Equivalently, the 5D reachable set over-approximated by the
%         intersection of backprojection of subsystem reachable sets
%     4c. Equivalently, a point (x, y, \theta, v, \omega) is in the
%         over-approximation if the state variables evaluate to a negative
%         value on all subsystem value functions
%
% Mo Chen, 2018-05-04

global gXT gYT gV gW; % Gotta make this usable from other functions.

%% Target and obstacles
% 5D grid limits (x, y, \theta, v, \omega)
if nargin < 3
  gMin = [-5; -5; -pi; 0; -1];
  gMax = [5; 5; pi; 3; 1];
  gN = 101*ones(5,1);
end

XTdims = [1 3];
YTdims = [2 3];
Vdim = 4;
Wdim = 5;

% Create grid structures for computation
% gXT = createGrid(gMin(XTdims), gMax(XTdims), gN(XTdims), 2);
% gYT = createGrid(gMin(YTdims), gMax(YTdims), gN(YTdims), 2);
% gV = createGrid(gMin(Vdim), gMax(Vdim), gN(Vdim), 2);
% gW = createGrid(gMin(Wdim), gMax(Wdim), gN(Wdim), 2);

%% Plane parameters
if nargin < 7
  aMax = 1;     % max. linear acceleration
  alphaMax = 1; % max. angular acceleration
  
  wMax = 0.5;   % max. turn rate for (x, \theta) and (y, \theta) subsystems
  vRange = [1; 2]; % speed range for (x, \theta) and (y, \theta) subsystems
end

%% Time parameters
if nargin < 8
  tMax = 1;
end

 % Time horizon and intermediate results
tau = 0:0.1:tMax;

plXT = Plane5D_XT([], vRange, wMax); % Create (x, \theta) subsystem
plYT = Plane5D_YT([], vRange, wMax); % Create (x, \theta) subsystem
plV = KinVehicleND(0, aMax);         % Create v subsystem
plW = KinVehicleND(0, alphaMax);     % Create (x, \theta) subsystem

%% Default target set
if nargin < 12
  targetXT = shapeCylinder(gXT, [], [0; 0], 1);
  targetYT = shapeCylinder(gYT, [], [0; 0], 1);
  targetV = shapeCylinder(gV, [], 1.5, 0.1);
  targetW = shapeCylinder(gW, [], 0, 0.05);
end

%% Compute reachable set
% Solver parameters
uMode = 'min';
dMode = 'min';
vis = false;
quiet = true;
keepLast = false;

sDXT.dynSys = plXT;
sDXT.grid = gXT;
sDXT.uMode = uMode;
sDXT.dMode = dMode;
eAXT.visualize = vis;
eAXT.quiet = quiet;
eAXT.keepLast = keepLast;

sDYT.dynSys = plYT;
sDYT.grid = gYT;
sDYT.uMode = uMode;
sDYT.dMode = dMode;
eAYT.visualize = vis;
eAYT.quiet = quiet;
eAYT.keepLast = keepLast;

sDV.dynSys = plV;
sDV.grid = gV;
sDV.uMode = uMode;
sDV.dMode = dMode;
eAV.visualize = vis;
eAV.quiet = quiet;
eAV.keepLast = keepLast;

sDW.dynSys = plW;
sDW.grid = gW;
sDW.uMode = uMode;
sDW.dMode = dMode;
eAW.visualize = vis;
eAW.quiet = quiet;
eAW.keepLast = keepLast;

% Call solver
dataXT = HJIPDE_solve(targetXT, tau, sDXT, 'none', eAXT);
dataYT = HJIPDE_solve(targetYT, tau, sDYT, 'none', eAYT);
dataV = HJIPDE_solve(targetV, tau, sDV, 'none', eAV);
dataW = HJIPDE_solve(targetW, tau, sDW, 'none', eAW);

end
