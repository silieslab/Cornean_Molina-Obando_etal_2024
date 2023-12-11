%% Create a figure big enough to fill almost whole (secondary) monitor.
function hFig = createFullScreenFig
graphicsRoot = groot;

if size(graphicsRoot.MonitorPositions, 1) == 2
    monitorPos = graphicsRoot.MonitorPositions(2, :);
elseif size(graphicsRoot.MonitorPositions, 1) == 1
    monitorPos = graphicsRoot.MonitorPositions(1, :);
end

figPos = [monitorPos(1:2) + 10 (monitorPos(3:4) - 100)];

hFig = figure('Position', figPos); % My pc.