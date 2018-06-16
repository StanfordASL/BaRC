function C = getContour(g, data, brs_idx, level)
% C = getContour(g, data, brs_idx, level)
%
% Inputs: g          - grid structure
%         data       - value function corresponding to grid g
%         level      - level set to display (defaults to 0)
%         brs_idx    - which backreachable output to use (timestep slice)
%
% Output: C - ContourMatrix
%
% Boris Ivanovic, 2018-05-23

%% Default parameters and input check
if isempty(g)
  N = size(data)';
  g = createGrid(ones(numDims(data), 1), N, N);
end

if g.dim ~= numDims(data) && g.dim+1 ~= numDims(data)
  error('Grid dimension is inconsistent with data dimension!')
end

%% Defaults
if nargin < 4
  level = 0;
end

C = contourc(g.vs{1}, g.vs{2}, data(:, :, brs_idx)', [level level]);

end
