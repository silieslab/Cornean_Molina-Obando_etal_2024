function [hFig, hAx, hScatter, hLine, hPatch] = plotScatterLineFitPredInts(x, y)

[xSorted, sortInd] = sort(x);
lineFit = fit(xSorted', y(sortInd)', 'poly1');

confIntType = 'functional';
confIntSimultaneous = 'on';
confLevelArray = [.85, 0.95, 0.99];

hFig = createFullScreenFig;
hAx = gca;
hold on;
lineColor = [.9 0.4 0.7];
patchColor = [.9 0.4 0.7];
patchAlpha =  0.2;
for iLevel = confLevelArray
    predInt = range(predint(lineFit, xSorted, iLevel, confIntType, confIntSimultaneous), 2) / 2;
    [hLine, hPatch] = plotErrorPatch(hAx, xSorted, lineFit(xSorted), predInt, lineColor, patchColor, patchAlpha);
    % shadedErrorBar(xSorted, lineFit(xSorted), predInt, {'Color', patchColor}, patchAlpha);
    % plot(xSorted, smooth(xSorted, y(sortInd), 0.5, 'rloess'))
end
hScatter = scatter(xSorted, y(sortInd), 50,  'filled', 'MarkerFaceAlpha', 0.2);

end