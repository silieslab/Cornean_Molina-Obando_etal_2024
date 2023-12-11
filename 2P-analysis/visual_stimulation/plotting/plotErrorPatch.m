function [hLine, hErrorPatch] = plotErrorPatch(hAx, x, y, error, lineColor, patchColor, varargin)

if nargin >= 7
    faceAlpha = varargin{1};
else
    faceAlpha = 0.5;
end

x=x(:)';
y=y(:)';
error = error(:)';

yUpper = y + error;
yLower = y - error; 
yLower = yLower(end:-1:1); % Flip lower bound coordinates to use in pacth function.
yPatch = [yUpper, yLower];
xPatch = [x, x(end:-1:1)]; % Flip x coordinates to use in pacth function.
hold(hAx, 'on');
hErrorPatch = patch(hAx, xPatch, yPatch, patchColor, 'linestyle', 'none', 'FaceAlpha', faceAlpha);
hLine = plot(hAx, x, y, 'color', lineColor);

end