function plotPCAExpVarianceAndTwoPCs(pcVectors, variances, pcaCoords, groupInds, figDir, nameSuffix)

hFig = createPrintFig(14 * [1 1]);
nSubPlots = 4;
hSubAx = createSquarishSubplotGrid(nSubPlots, [0.1 0.12]);

varColors = brewermap(2, 'OrRd');
axes(hSubAx(1));
bar(cumsum(variances), 'EdgeColor', 'none', 'FaceColor', varColors(1, :));
hold on;
bar(variances, 'EdgeColor', 'none', 'FaceColor', varColors(2, :));
title(gca, 'Explained variance (%)')
xlabel('Principal components')
axis square tight

axes(hSubAx(2));
pcColorMap = brewermap([], 'YlOrRd');
% pcColorMap = flipud(parula);
colors = linearlyMapArrayToColors(variances, pcColorMap);
hPCLines = plot((0: size(pcVectors, 1) - 1) / 10, bsxfun(@plus, pcVectors(:, 1: 5), -max(pcVectors(:)) * (1: 5)));
for iLine = 1: numel(hPCLines)
    hPCLines(iLine).Color = colors(iLine, :);
end
hSubAx(2).YAxis.Visible = 'off';
xlabel('Time (s)')
title('Principal components')
axis square tight
axes(hSubAx(3));
% Get PCA coords.
colorMap = brewermap(256, 'YlGnBu');
colorMap = flipud(colorMap);
% colorMap = colorMap(1: floor(0.7*size(colorMap, 1)), :);
colormap(colorMap);
hScatt = scatter(pcaCoords(:, 1), pcaCoords(:, 2), [], groupInds, 'filled', ...
                 'MArkerFaceAlpha', 1, 'MarkerEdgeColor', 'k');
xlabel(['First PC (' num2str(variances(1), 2) '%)'])
ylabel(['Second PC (' num2str(variances(2), 2) '%)'])
title('Principal components 1 vs 2')
axis square

axes(hSubAx(4));
colormap(colorMap)
hScatt = scatter(pcaCoords(:, 1), pcaCoords(:, 3), [], groupInds, 'filled', ...
                 'MArkerFaceAlpha', 1,  'MarkerEdgeColor', 'k');
xlabel(['First PC (' num2str(variances(1), 2) '%)'])
ylabel(['Third PC (' num2str(variances(3), 2) '%)'])
title('Principal components 1 vs 3')
originPos = hSubAx(4).Position;
hCbar = colorbar;
hSubAx(4).Position = originPos;
axis square

arrayfun(@prettifyAxes, hSubAx);
arrayfun(@offsetAxes, hSubAx);
setFontForThesis(hSubAx, gcf)

figFileName = ['OnOff-PCA' nameSuffix];
% set(gcf,'renderer','painters')
% set(gcf, 'PaperPositionMode', 'auto');
print(hFig, [figDir figFileName '.pdf'], '-dpdf')
print(hFig, [figDir figFileName '.png'], '-dpng', '-r300')

end