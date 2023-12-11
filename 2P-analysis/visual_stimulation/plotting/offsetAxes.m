function offsetAxes(ax)
% thanks to Pierre Morel, undocumented Matlab
% and https://stackoverflow.com/questions/38255048/separating-axes-from-plot-area-in-matlab
%
% by Anne Urai, 2016
% Modified Luis Ramos T. 2019
if ~exist('ax', 'var'), ax = gca; end
% If there is only one tick do not change stuff.
% Otherwise modify the x and y limits to below the data (by a small amount)
if numel(ax.XTick) == 1 && numel(ax.YTick) == 1
    return
end

if numel(ax.XTick) > 1
    shiftXAx = (ax.XTick(2) - ax.XTick(1)) / 4;
    ax.XLim = ax.XLim + [-1 1] * shiftXAx;
end
if numel(ax.YTick) > 1
    shiftYAx = (ax.YTick(2) - ax.YTick(1)) / 4;
    ax.YLim = ax.YLim + [-1 1] * shiftYAx;
end

% this will keep the changes constant even when resizing axes
switch ax.Type
    case 'colorbar'
        return
    otherwise
        addlistener (ax, 'MarkedClean', @(obj,event)resetVertex(ax));
end
end

function resetVertex ( ax )
% extract the x axis vertext data
% X, Y and Z row of the start and end of the individual axle.
ax.XRuler.Axle.VertexData(1,1) = min(get(ax, 'Xtick'));
% repeat for Y (set 2nd row)
ax.YRuler.Axle.VertexData(2,1) = min(get(ax, 'Ytick'));
% Limit last tick
% X, Y and Z row of the start and end of the individual axle.
ax.XRuler.Axle.VertexData(1,2) = max(get(ax, 'Xtick'));
% repeat for Y (set 2nd row)
ax.YRuler.Axle.VertexData(2,2) = max(get(ax, 'Ytick'));
end