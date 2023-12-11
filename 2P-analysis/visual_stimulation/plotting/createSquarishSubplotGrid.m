function hSubplotAxArray  = createSquarishSubplotGrid(nSubplots, varargin)

if nargin > 1
    spacing = varargin{1};
else
    spacing = [0 0];
end
% Closest integer to square root of number of needed subplots.
nRows = ceil(sqrt(nSubplots));
% Now just add the needed colums, here maybe we end up with an extra
% column.
nCols = ceil(nSubplots / nRows);

hSubplotAxArray = arrayfun(@(x) subplot_tight(nRows, nCols, x, spacing), 1: nSubplots);

end