close all;
clear all;
clc;

addpath(genpath(pwd))

%% Define search path.
rootDir = pwd;
dataFolder = fullfile(rootDir, 'data');
dataFile = 'Fig-Tm9-processedTableWithBackgroundOnOff.mat';
dataPath = fullfile(dataFolder, dataFile);
load(dataPath, 'onoffTable');

%%
figDir = fullfile(rootDir, 'figures');
if ~exist(figDir, 'dir'); mkdir(figDir); end


%%
% Extract the fullfield data
stimSetCellStr = getStimSetCellStr(1);
onoffTable = getStimSubsetTable(onoffTable, stimSetCellStr, 1);
onoffTracesPerStackCell = arrayfun(@(x) cat(1, x{1}{:}), onoffTable.paddedEpochStacks, 'UniformOutput', false);
onoffTracesPerROIArray = cat(2, onoffTracesPerStackCell{:});

% extract response index as a vector
responseIndexArray = [onoffTable.responseIndex{:}];
% extract meanOFF vector
meanOFFCell = arrayfun(@(x) cat(1, x{1}.offMean), onoffTable.paramTuning, 'UniformOutput', false);
meanOFFArray = cat(2, meanOFFCell{:});
% extract transient index
sustenanceOFFCell = arrayfun(@(x) cat(1, x{1}.lateToExtremeRatioOFF), onoffTable.paramTuning, 'UniformOutput', false);
sustenanceOFFArray = cat(2, sustenanceOFFCell{:});
% extract polarity index
polarityIndexCell = arrayfun(@(x) cat(1, x{1}.polarityIndex), onoffTable.paramTuning, 'UniformOutput', false);
polarityIndexArray = cat(2, polarityIndexCell{:});


%%
rng default;

zScoredTraces = bsxfun(@rdivide, onoffTracesPerROIArray - mean(onoffTracesPerROIArray, 1), std(onoffTracesPerROIArray, [], 1));
[U, S, V, variances, pcaCoords, dataReduced] = doPCA(zScoredTraces);
tsneVec = tsne(zScoredTraces');

% clustering
nClusters = 6;

rng default;
clusterInds = kmeans(onoffTracesPerROIArray', nClusters, Replicates=20, Distance="correlation");
nClusters = numel(unique(clusterInds));
% clusterInds = clusterdata(onoffTracesPerROIArray', 'maxclust', 3);
clusterTracesCell = arrayfun(@(x) zScoredTraces(:, clusterInds == x), 1: nClusters, UniformOutput=false);
% clusterTracesCell = arrayfun(@(x) onoffTracesPerROIArray(:, clusterInds == x), 1: nClusters, UniformOutput=false)

%%
% evaluate the response quality per cluster
clusterRespIndex = accumarray(clusterInds, responseIndexArray, [], @mean);
% clusterRespIndex = accumarray(clusterInds, meanOFFArray, [], @mean);
clusterRespIndex = accumarray(clusterInds, sustenanceOFFArray, [], @mean);
% clusterRespIndex = accumarray(clusterInds, polarityIndexArray, [], @mean);
% sort clusters
[~, clusterSortInds] = sort(clusterRespIndex, "descend");

for ind = 1: numel(clusterInds)
    clusterInds(ind) = clusterSortInds(clusterInds(ind));
end
%%
blueColor = lines(1);
lab = rgb2lab(blueColor);
lighterBlue = lab2rgb(lab .* [1.3 1 1]);

figure('WindowStyle','docked');
clf
h(1) = subplot(2,3,1);
plot(onoffTracesPerROIArray, Color=[lighterBlue 0.1]);
h(1).Title.String = 'ONOFF traces dF/F0';
hold on;
plot(mean(onoffTracesPerROIArray, 2), LineWidth=3, Color=blueColor)
h(2) = subplot(2,3,2);
plot(zScoredTraces, Color=[lighterBlue 0.1]);
h(2).Title.String = 'ONOFF traces zScored';
h(4) = subplot(2,3,4);
scatter(V(:, 1), V(:, 2), 'o', 'filled', 'MarkerFaceAlpha', 0.5);
h(4).Title.String = 'PCA of zScored traces';
h(5) = subplot(2,3,5);
colormap(lines(nClusters));
scatter(V(:, 1), V(:, 2), 50, clusterInds, 'o', 'filled', 'MarkerFaceAlpha', 1);
h(5).Title.String = {'PCA colored by cluster ID'};

h(3) = subplot(2,3,3);
colorMap = lines(nClusters);
% for iLine = 1: size(zScoredTraces, 2)
%     plot(1: size(zScoredTraces, 1), zScoredTraces(:, iLine) + ...
%         max(zScoredTraces(:)) * clusterInds(iLine), ...
%         Color=[colorMap(clusterInds(iLine), :) 0.1])
%     hold on;
% end
for iCluster = 1: nClusters
    plot(1 : size(clusterTracesCell{iCluster}, 1), ...
        clusterTracesCell{iCluster} + 1.1 * max(zScoredTraces(:)) * iCluster, ...
        Color=[colorMap(iCluster, :) 0.1], LineWidth=0.5);
    hold on;
    plot(1 : size(clusterTracesCell{iCluster}, 1), ...
        mean(clusterTracesCell{iCluster}, 2)+ 1.1 * max(zScoredTraces(:)) * iCluster, ...
        Color=[colorMap(iCluster, :) 1], LineWidth=2);
end
h(3).Title.String = 'zScored traces by cluster';
h(6) = subplot(2,3,6);
% colormap("lines")
scatter(tsneVec(:, 1), tsneVec(:, 2), 50, clusterInds, "filled");
h(6).Title.String = 'tSNE of zScored traces';

axis(h, "tight");
arrayfun(@prettifyAxes, h);
print('variability-onoff-fullfield-Tm9', '-dpdf', '-vector');

