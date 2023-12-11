function plotPCAwithRespQuality(pcaCoords, responseInd, respThreshold)
qualityInds = responseInd > respThreshold;
figure
scatter(pcaCoords(qualityInds, 1), pcaCoords(qualityInds, 2), [], responseInd(qualityInds), 'filled')
caxis([0 1])
setFavoriteColormap
hCBar = colorbar;
xlabel('First PC');
ylabel('Second PC');
hCBar.Label.String = 'Resp. Quality Index';
arrayfun(@prettifyAxes, [gca hCBar])
end