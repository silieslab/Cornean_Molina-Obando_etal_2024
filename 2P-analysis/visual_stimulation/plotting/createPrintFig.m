function hFig = createPrintFig(figSize)
% Defaults for this fig.
width = figSize(1);     % Width in centimeters
height = figSize(2);    % Height in centimeters
alw = 0.75;             % AxesLineWidth
fsz = 11;               % Fontsize
lw = 2;                 % LineWidth
msz = 6;                % MarkerSize


graphicsRoot = groot;

if size(graphicsRoot.MonitorPositions, 1) == 2
    monitorPos = graphicsRoot.MonitorPositions(2, :);
elseif size(graphicsRoot.MonitorPositions, 1) == 1
    monitorPos = graphicsRoot.MonitorPositions(1, :);
end

defaultPos = [monitorPos(1:2) + 10 (monitorPos(3:4) - 100)];

% The properties we've been using in the figures
set(0, 'defaultLineLineWidth', lw);   % set the default line width to lw
set(0, 'defaultLineMarkerSize', msz); % set the default line marker size to msz

%% Set the default Size for display
% defaultPos = figPos;% get(0, 'defaultFigurePosition');
% set(0, 'defaultFigurePosition', [defaultPos(1), defaultPos(2), width * 100, height * 100]);
hFig = figure;
hFig.Position = [defaultPos(1), defaultPos(2), ...
                 min(width * 75, defaultPos(3)) min(height * 75, defaultPos(4))];
%% Set the defaults for saving/printing to a file
% set(0, 'defaultFigureInvertHardcopy', 'on'); % This is the default anyway
% set(0, 'defaultFigurePaperUnits', 'centimeters'); % This is the default anyway
hFig.InvertHardcopy = 'on';
hFig.PaperUnits = 'centimeters';
if 0
    defaultPaperSize = get(0, 'defaultFigurePaperSize');
else
    hFig.PaperSize = [width height];
    defaultPaperSize = [width height];
end
left = (defaultPaperSize(1) - width) / 2;
bottom = (defaultPaperSize(2) - height) / 2;
% defaultPaperSize = [left, bottom, width, height];
% set(0, 'defaultFigurePaperPosition', defaultPaperSize);
hFig.PaperPosition = [left, bottom, width, height];

end