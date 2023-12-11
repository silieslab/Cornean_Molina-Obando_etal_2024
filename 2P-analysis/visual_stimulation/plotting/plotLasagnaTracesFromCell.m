
function hSubAx = plotLasagnaTracesFromCell(hFig, allNeuronsOnOffTraces, colors, varargin)
timePoints = (0: size(allNeuronsOnOffTraces{1}, 2)-1) /10;
if ~isempty(varargin)
    timePoints = varargin{1};
end
cLims = ZeroCenteredBounds(cat(1, allNeuronsOnOffTraces{:}), [0 100]);
figure(hFig)
nNeurons = numel(allNeuronsOnOffTraces);
for iNeuron = 1: nNeurons
    hSubAx(iNeuron, 1) = subplot(3, nNeurons + 1, iNeuron);
    hSubAx(iNeuron, 2) = subplot(3, nNeurons + 1, iNeuron + (nNeurons + 1) * (1:2));
    axes(hSubAx(iNeuron, 1))
    hold on;
    plot(timePoints, allNeuronsOnOffTraces{iNeuron}', 'Color', [colors(iNeuron, :) 0.3], 'LineWidth', 1);
    plot(timePoints, mean(allNeuronsOnOffTraces{iNeuron}), 'Color', [colors(iNeuron, :)], 'LineWidth', 3);
%     shadedErrorBar(timePoints, allNeuronsOnOffTraces{iNeuron}, {@mean @(x) std(x) / sqrt(size(x, 1))}, {'Color', [colors(iNeuron, :) 0.3], 'LineWidth', 3});
    axis tight
    if iNeuron > 1
        hSubAx(iNeuron, 1).YAxis.Visible = 'off';
    end
    hSubAx(iNeuron, 1).XAxis.Visible = 'off';
    hSubAx(iNeuron, 1).Position(2) = sum(hSubAx(iNeuron, 2).Position([2, 4]));
    
    axes(hSubAx(iNeuron, 2))
%     cLims = ZeroCenteredBounds(cat(1, allNeuronsOnOffTraces{:}), [0 100]);
    imagesc(timePoints, [], allNeuronsOnOffTraces{iNeuron}, cLims);
    setFavoriteColormap;
    imAxisRatio = hSubAx(iNeuron, 2).PlotBoxAspectRatio(1) / hSubAx(iNeuron, 2).PlotBoxAspectRatio(2);
    xAxisRatio = imAxisRatio * hSubAx(iNeuron, 2).Position(4) / hSubAx(iNeuron, 1).Position(4);
%     hSubAx(iNeuron, 1).PlotBoxAspectRatio = [xAxisRatio, 1, 1];
    pos1 = hSubAx(iNeuron, 1).Position;
    pos2 = hSubAx(iNeuron, 2).Position;
    pos1(3) = pos2(3);
    set(hSubAx(iNeuron, 1), 'Position', pos1);
    hSubAx(iNeuron, 1).Position(3) = hSubAx(iNeuron, 2).Position(3);
    
end


    
end