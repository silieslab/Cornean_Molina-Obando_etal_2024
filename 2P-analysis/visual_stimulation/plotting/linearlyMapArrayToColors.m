function colors = linearlyMapArrayToColors(xArray, colorMap)

% Normalize to range 0-1.
if ~range(xArray(:)) == 0
    xArray = (xArray - min(xArray(:))) / range(xArray(:));
end
nColors = size(colorMap, 1);
colorInds = floor(xArray * (nColors - 1)) + 1;
colors = colorMap(colorInds, :);

end