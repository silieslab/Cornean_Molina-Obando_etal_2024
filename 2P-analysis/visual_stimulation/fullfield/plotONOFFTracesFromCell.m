
function plotONOFFTracesFromCell(allNeuronsOnOffTraces, cellTypeStr, figDir, colors)
cLims = ZeroCenteredBounds(cat(1, allNeuronsOnOffTraces{:}), [0 100]);
hFig = createPrintFig(13 * [1 1]);
hSubAx = plotLasagnaTracesFromCell(hFig, allNeuronsOnOffTraces, colors);
nNeurons = numel(allNeuronsOnOffTraces);

ylabel(hSubAx(1, 1), '\DeltaF / F_0');
title(hSubAx(1, 1), 'ON-OFF full field flashes')
ylabel(hSubAx(1, 2), 'ROI#');
xlabel(hSubAx(1, 2), 'Time (s)');
linkaxes([hSubAx(:, 1)], 'y')    
linkaxes([hSubAx(:, 2)], 'y')    

hAxDummy = subplot(3, nNeurons + 1, 2*(nNeurons + 1));
hAxDummy.Position(3: 4) = hAxDummy.Position(3: 4) / 2;
imagesc(rand(10)*0, cLims)
hAxDummy.Visible = 'off';
setFavoriteColormap
hCBar = colorbar;
hCBar.Limits = cLims;
hCBar.Label.String = '\Delta F / F_0';

linkaxes(hSubAx, 'x')
arrayfun(@prettifyAxes, [hSubAx(:); hCBar]);
arrayfun(@offsetAxes, [hSubAx(:); hCBar]);
hSubAx(1).XLim =[0 12];
setFontForThesis(hSubAx, hFig)

 figFileName = ['OnOff-Traces-' cellTypeStr{:}];
    set(gcf,'renderer','painters')
%     set(gcf, 'PaperPositionMode', 'auto');
    print(hFig, [figDir figFileName '.pdf'], '-dpdf')
    print(hFig, [figDir figFileName '.png'], '-dpng', '-r300')
    
end