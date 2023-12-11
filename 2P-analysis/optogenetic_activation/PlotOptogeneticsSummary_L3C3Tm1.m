ccc;
saveFolder = 'C:\Users\gio\Documents\MATLAB';
mainFolder = fullfile(saveFolder, 'Two_photonMat');


collectTimeSeries = false;
if collectTimeSeries

    %%
    cd(mainFolder)
    [~,list] = system('dir /S/B *CsChrimson');
    list = textscan(list, '%s', 'Delimiter', '\n');
    expFolders = list{1};
    
    l = 1;
    timeSeriesPaths = cell(1,500);
    for jExpFolder = 1: numel(expFolders)
        timeSeriesFolders = dir(expFolders{jExpFolder});
        timeSeriesFolders(1:2) = [];
        for kTimeSeries = 1: numel(timeSeriesFolders)
                    timeSeriesPaths{l} = [expFolders{jExpFolder} filesep ...
                                          timeSeriesFolders(kTimeSeries).name];
                    l = l + 1;
        end
    end
    
    %%
    emptyCells = cellfun(@(x) isempty(x), timeSeriesPaths);
    timeSeriesPaths(emptyCells) = [];
    nTimeSeries = numel(timeSeriesPaths);
    indsToDelete = zeros(nTimeSeries, 1);
    
    %%
    selectedPaths = timeSeriesPaths(contains(timeSeriesPaths, 'L3'));
    for iFolder = 1: numel(selectedPaths)
        xmlFiles = dir([selectedPaths{iFolder} filesep '*xml']);
        xmlPaths = arrayfun(@(x) fullfile(x.folder, x.name), xmlFiles, 'UniformOutput', false);
        % copy to the TwoPhoton folder
        % make new folder
        rawFolder = regexprep(selectedPaths{iFolder}, 'Two_photonMat', 'Two_photon');
        if ~exist(rawFolder, "dir"); mkdir(rawFolder); end
    
        cellfun(@(x) copyfile(x, regexprep(x, 'Two_photonMat', 'Two_photon')), xmlPaths)
    end
    %% Find the stack index corresponding to each time series folder
    %  for a given fly for a given experiment in a particular day.
    
    nTimeSeries = numel(timeSeriesPaths);
    dataPath = cell(nTimeSeries, 3);
    
    for jTimeSeries = 1: nTimeSeries 
        [basePath, timeSeriesFolder] = fileparts(timeSeriesPaths{jTimeSeries});
        [basePath, ~, ext] = fileparts(basePath);
        flyInd = regexp(ext, 'fly(\d*)', 'tokens');
        flyInd = str2double(cell2mat(flyInd{:}));
    %     flyInd = str2double(ext(strfind(ext, 'fly') + 3));
        
        search_string = fullfile(basePath, sprintf('*fly%d*', flyInd));
        listing = dir(search_string);
        flyFolder =  fullfile(basePath, listing(1).name);
        search_stack_str = fullfile(flyFolder, '*');
        stackListing = dir(search_stack_str);
        % Delete the '.' and '..' results of dir.
        invalidFileIdx = cell2mat(cellfun(@(x) strcmp(x,'.') || ...
                                  strcmp(x,'..') || ...
                                  ~isempty(strfind(x, 'SingleImage')) || ...
                                  ~isempty(strfind(x, 'ZSeries')) || ...
                                  ~isempty(strfind(x, 'BrightnessOverTime')), ...
                                  {stackListing.name},'UniformOutput', false));
        stackListing(invalidFileIdx) = [];
        stackListing(~[stackListing.isdir]) = [];
        stackInd = cell2mat(cellfun(@(x) strcmp(x, timeSeriesFolder), ...
                                    {stackListing.name},'UniformOutput', false));
        stackInd = find(stackInd);
        dataPath{jTimeSeries, 1} = basePath;
        dataPath{jTimeSeries, 2} = flyInd;
        dataPath{jTimeSeries, 3} = stackInd;
    end
    
    
    
    %% Load the corresponding data.
    timeSeriesCollection(nTimeSeries).stimulus = [];
    timeSeriesCollection(nTimeSeries).data = [];
    timeSeriesCollection(nTimeSeries).dataPath = [];
    
    useImagingRoiStruct = false;
    uiSelection = false;
    
    %%
    for kTimeSeries = 1: nTimeSeries
        
        basePath = dataPath{kTimeSeries, 1};
        
        flyInd = dataPath{kTimeSeries, 2};
        stackInd = dataPath{kTimeSeries, 3};
        
        baseName = FindSessionPrefix(basePath, flyInd, stackInd);
    %     stackInd = 1;
    
        imagingStructFilename = [baseName filesep '_imagingStruct.mat'];
        f = matfile(imagingStructFilename);
        [nRows, nCols] = size(f, 'imagingStruct');
        if nCols > 1 || strcmp(basePath, 'D:\Data_LGRTL\Two_photonMat\161127.0.lr')
            continue;
        end
    %     if ~isempty( stackInd )
    %         out = f.imagingStruct(1, stackInd);
    %     else
        out = f.imagingStruct;
    %     end
        imagingStruct = out;
        clear out;
        
        xmlFilePath = dir([regexprep(baseName, 'Two_photonMat', 'Two_photon') ...
                           filesep '*Mark*xml']);
        xmlFilePath = [regexprep(baseName, 'Two_photonMat', 'Two_photon') ...
                           filesep xmlFilePath.name];
        
        % Load existent roiStruct or select ROIs, do not choose overlapping regions.
        imagingStruct(1).experimentParentDir = regexprep(mainFolder, 'Two_photonMat', 'Two_photon');     % modified by Luis 28.11.2023
        roiStruct = selectOrLoadROIs( basePath, flyInd, stackInd, ...
                                      imagingStruct, uiSelection );
        imStack = imagingStruct.stackRaw;
        roiTCs = ApplyROIsToStack_TimeCourse(imStack, roiStruct);
        roiTCs = reshape(roiTCs, [1 size(roiTCs)]);
    %     roiTCsMinusBackground = bsxfun(@minus, roiTCs, roiTCs(1, end, :)); 
    %     roiTCs = roiTCsMinusBackground(:, 1: end - 1, :);
    
    %     roiTCsNormalized = NormalizeStack( roiTCs, taTimingStruct, 'dF_F0' );
    %     Compute correlation coefficient.
        yAll = squeeze(roiTCs);
        timeSeriesCollection(kTimeSeries).stimulus = parseXmlOptoStim(xmlFilePath);
        timeSeriesCollection(kTimeSeries).data = yAll;
        timeSeriesCollection(kTimeSeries).dataPath = dataPath(kTimeSeries, :);
        timeSeriesCollection(kTimeSeries).frameDuration = imagingStruct.frameDuration;
        clear imagingStruct;
    
    end
    %%
    saveToFileAgain = False;
    if saveToFileAgain
        optoFileName = '170928_lr_L3_C3_Tm1_OptogeneticsWoNaNsWoBGSubstraction.mat';
        timeSeriesFilePath = fullfile(saveFolder, optoFileName);
        save(timeSeriesFilePath, 'timeSeriesCollection', '-v7.3');
    end
end
%%

optoFileName = '170928_lr_L3_C3_Tm1_OptogeneticsWoNaNsWoBGSubstraction.mat';
load( fullfile(saveFolder, optoFileName), 'timeSeriesCollection')

%% Valid time series
% Time series with 2000 frames.
nFramesArray = arrayfun(@(x) size(x.data,2), timeSeriesCollection);
invalidTimeSeriesIdx = find(nFramesArray ~= 2000);
validTimeSeriesCollection = timeSeriesCollection;
validTimeSeriesCollection(invalidTimeSeriesIdx) = [];

%% Interpolate signals to 10 Hz.
endTimes = arrayfun(@(x) mode(nFramesArray) * x.frameDuration, ...
                    validTimeSeriesCollection);
targetFreq = 10;
targetTimes = 1 / targetFreq : 1 / targetFreq : round(min(endTimes), 1) + ...
              1 / 2 / targetFreq;
targetTimes = targetTimes - 1 / 2 / targetFreq;
interpSeries{numel(validTimeSeriesCollection)} = [];
for iStack = 1: numel(validTimeSeriesCollection)
    interpSeries{iStack} = interp1( validTimeSeriesCollection(iStack). ...
                                    frameDuration * (1:mode(nFramesArray)), ...
                                    validTimeSeriesCollection(iStack).data', ...
                                    targetTimes, 'linear', 'extrap' );
    validTimeSeriesCollection(iStack).interpData = ...
        interp1( validTimeSeriesCollection(iStack). ...
                 frameDuration * (1:mode(nFramesArray)), ...
                 validTimeSeriesCollection(iStack).data', ...
                 targetTimes, 'linear', 'extrap' );
    
end

%% replace the original folder to the current one
originalFolder = 'D:\Data_LGRTL';
for iSeries = 1: numel(validTimeSeriesCollection)
    validTimeSeriesCollection(iSeries).dataPath{1} = ...
        strrep(validTimeSeriesCollection(iSeries).dataPath{1}, ...
               originalFolder, saveFolder);
end
%% Prepare for plotting
close all
minValue = inf;
maxValue = -inf;
genotypeCell = {'L3', 'C3', 'Tm1'};
stimCell = {'25pulses_5ms_40Hz_30sISI_5reps_100LED'};
processedROIs = cell(numel(genotypeCell), numel(stimCell));

for iGenotype = 1: numel(genotypeCell)
    genotypeStr = genotypeCell{iGenotype};
    for jStimulus = 1: numel(stimCell)
        stimStr = stimCell{jStimulus};
        titleStr = ['Tm9GCaMP6f-' genotypeStr 'CsChrimson: ' stimStr '_nonNaN'];
        exclusionCriteria = 'max((trialAverage)) < -20 || any(isnan(trialAverage))';
        [genotypeStimArray, nFlies, ...
         genotypeInds, stimInds] = getGenotypeStimArray(validTimeSeriesCollection, ...
                                                           genotypeStr, stimStr);
        trialAveragedNormalizedROIs = trialAverageAndNormalizeArray(genotypeStimArray, targetTimes, stimStr);
%         plotROIsLasagnaAndMean(trialAveragedNormalizedROIs, targetTimes, titleStr, nFlies)
        minValue = min(minValue, mean(trialAveragedNormalizedROIs(:)) - 3 * std(trialAveragedNormalizedROIs(:)));
        maxValue = max(maxValue, mean(trialAveragedNormalizedROIs(:)) + 3 * std(trialAveragedNormalizedROIs(:)));
        processedROIs{iGenotype, jStimulus} = trialAveragedNormalizedROIs;
        genotypeStimInds{iGenotype, jStimulus} = stimInds;
    end
end


%%
figure
colors = [162,144,182;
         231,47,143;
         228,64,55] / 255;


genotypeCell = {'L3', 'C3', 'Tm1'};
desiredGenotypes = {'L3', 'C3', 'Tm1'};
desiredGenotypeInds = cellfun(@(x) find(strcmp(x, genotypeCell)), desiredGenotypes);

iPlot = 1;
stimInd = 1;
for iGenotype = desiredGenotypeInds
    hSub(iGenotype) = subplot_tight(1, 3, iPlot, 0.05 * [1 1]);
%     hSub(iGenotype) = subplot_tight(3, 3, iPlot, 0.05 * [1 1]);
    hold on;
    genotypeStr = genotypeCell{iGenotype};
    stimStr = stimCell{stimInd};
    [~, nFlies] = getGenotypeStimArray(validTimeSeriesCollection, ...
                                       genotypeStr, stimStr);
    trialAveragedNormalizedROIs = processedROIs{iGenotype, stimInd};
    nROIs = sum(~isnan(mean(trialAveragedNormalizedROIs(:,1:2), 2)));
    jColor = iPlot;
    trialAveragedNormalizedROIs = processedROIs{iGenotype, stimInd};
    [nPulses, pulseDurMilliSec, ...
     ISIseconds, nTrials, ...
     ledPowerLEDunits, stimDur, freqHz] = parseStimStrOptogenetics(stimStr);
    nCells = size(trialAveragedNormalizedROIs, 1);
    if stimDur > mean(diff(targetTimes))
        stimStart = 10 - mean(diff(targetTimes)); 
        stimEnd = stimStart + stimDur + 2 * mean(diff(targetTimes));
    else
        stimStart = 10; 
        stimEnd = stimStart + stimDur;
    end
    if jStimulus == 1 || jStimulus == 6 && ~isempty(trialAveragedNormalizedROIs)
        xStim = [stimStart stimEnd stimEnd stimStart];
        yStim = [-0.65 * [1 1], 2.6 * [1 1]];
        hStimPatchCopy = patch(xStim, yStim, 'red');
        hStimPatchCopy.LineStyle = 'none';
        hStimPatchCopy.FaceAlpha = 0.5;
    end
    trialTimes = targetTimes(1:size(trialAveragedNormalizedROIs, 2));
    trialAveragedNormalizedROIs(:, trialTimes >= stimStart & trialTimes <= stimEnd) = nan;
    if strcmp(genotypeCell{iGenotype}, 'Dm4') 
        [~, peakInds] = max(abs(trialAveragedNormalizedROIs(:, trialTimes >= stimStart & trialTimes <= stimStart + 1)), [], 2);
        peakResponse = arrayfun(@(x) trialAveragedNormalizedROIs(x, find(trialTimes >= stimStart, 1) + peakInds(x)), 1:numel(peakInds));
        positiveTrialAveragedNormalizedROIs = trialAveragedNormalizedROIs(peakResponse > 0, :);
        negativeTrialAveragedNormalizedROIs = trialAveragedNormalizedROIs(peakResponse < 0, :);
        hRoiMeanPos = shadedErrorBar(trialTimes(trialTimes <= stimStart), positiveTrialAveragedNormalizedROIs(:, trialTimes < stimStart), ...
                  {@nanmean, @(x) 1*nanstd(x)/sqrt(sum(~isnan(mean(x, 2)))) ...
                  }, {'-', 'Color', colors(jColor, :), 'LineWidth', 2}, 0);
        hold on;
        hRoiMean2Pos = shadedErrorBar(trialTimes(trialTimes >= stimEnd), positiveTrialAveragedNormalizedROIs(:, trialTimes > stimEnd), ...
                          {@nanmean, @(x) 1*nanstd(x)/sqrt(sum(~isnan(mean(x, 2)))) ...
                          }, {'-', 'Color', colors(jColor, :), 'LineWidth', 2}, 0);       
        hRoiMeanNeg = shadedErrorBar(trialTimes(trialTimes <= stimStart), negativeTrialAveragedNormalizedROIs(:, trialTimes < stimStart), ...
                {@nanmean, @(x) 1*nanstd(x)/sqrt(sum(~isnan(mean(x, 2)))) ...
                }, {'-', 'Color', colors(jColor, :), 'LineWidth', 2}, 0);
        hold on;
        hRoiMean2Neg = shadedErrorBar(trialTimes(trialTimes >= stimEnd), negativeTrialAveragedNormalizedROIs(:, trialTimes > stimEnd), ...
                          {@nanmean, @(x) 1*nanstd(x)/sqrt(sum(~isnan(mean(x, 2)))) ...
                          }, {'-', 'Color', colors(jColor, :), 'LineWidth', 2}, 0);
        [hRoiMeanPos.edge(1).Annotation.LegendInformation.IconDisplayStyle] = deal('off');
        [hRoiMeanNeg.edge(1).Annotation.LegendInformation.IconDisplayStyle] = deal('off');
        [hRoiMeanPos.edge(2).Annotation.LegendInformation.IconDisplayStyle] = deal('off');
        [hRoiMeanNeg.edge(2).Annotation.LegendInformation.IconDisplayStyle] = deal('off');
        nROIsPos = size(positiveTrialAveragedNormalizedROIs, 1);
        nROIsNeg = size(negativeTrialAveragedNormalizedROIs, 1);
        [hMeanLegend, icons, plots, txt] = legend([hRoiMeanPos.mainLine hRoiMeanNeg.mainLine], ...
             [genotypeStr ' (n = ' num2str(nROIsPos) ', N = ' num2str(nFlies) ')' ], ...
             [genotypeStr ' (n = ' num2str(nROIsNeg) ', N = ' num2str(nFlies) ')' ], ...
             'Orientation', 'vertical', ...
             'Location', 'northeast', ...
             'Box', 'off');
         legend boxoff
    else
        hRoiMean = shadedErrorBar(trialTimes(trialTimes <= stimStart), trialAveragedNormalizedROIs(:, trialTimes < stimStart), ...
                          {@nanmean, @(x) 1*nanstd(x)/sqrt(sum(~isnan(mean(x, 2)))) ...
                          }, {'-', 'Color', colors(jColor, :), 'LineWidth', 2}, 0);
        hold on;
        hRoiMean2 = shadedErrorBar(trialTimes(trialTimes >= stimEnd), trialAveragedNormalizedROIs(:, trialTimes > stimEnd), ...
                          {@nanmean, @(x) 1*nanstd(x)/sqrt(sum(~isnan(mean(x, 2)))) ...
                          }, {'-', 'Color', colors(jColor, :), 'LineWidth', 2}, 0);
        [hRoiMean.edge(1).Annotation.LegendInformation.IconDisplayStyle] = deal('off');
        [hRoiMean.edge(2).Annotation.LegendInformation.IconDisplayStyle] = deal('off');
        [hMeanLegend, icons, plots, txt] = legend([hRoiMean.mainLine], ...
                     [genotypeStr ' (n = ' num2str(nROIs) ', N = ' num2str(nFlies) ')' ], ...
                     'Orientation', 'vertical', ...
                     'Location', 'northeast', ...
                     'Box', 'off');
        legend boxoff
    end
    icons(2).XData(1) = icons(2).XData(2) / 2;
    %     set(gca, 'ylim', [-0.65, 2.5]);
    %     plot(nanmean(processedROIs{iGenotype, jStimulus})); hold on

%     axis('square')
    axis tight
    hSub(iGenotype).PlotBoxAspectRatio = [4 3 1];
    hSub(iGenotype).XAxis.TickValues = [0 10];
    hSub(iGenotype).YAxis.TickValues = [-1 0 1] * 0.5;
    hSub(iGenotype).Color = 'none';
    hSub(iGenotype).TickDir = 'out';

    if iPlot ~= numel(desiredGenotypeInds)
        hSub(iGenotype).YAxis.Visible = 'off';
        hSub(iGenotype).XAxis.Visible = 'off';
    end
    iPlot = iPlot + 1;
end
 
saveFigureA4(gcf, [saveFolder filesep], 'OptogeneticsToTm9_L3-C3-Tm1', 300, 0, 'Landscape')

%%
% Save Optogenetics table for plotting in other places, 
% example python with pandas and seaborn

OptogeneticsTable = cell2table(processedROIs, "VariableNames", stimCell);
OptogeneticsTable(:, 'Genotype') = genotypeCell';

% varfun(@(x) cellfun(@(y) isempty(y), x), OptogeneticsTable)
columns = ["25pulses_5ms_40Hz_30sISI_5reps_100LED", "Genotype"];
rows = cellfun(@(x) find(strcmp(OptogeneticsTable.Genotype, x)), {'L3', 'C3', 'Tm1'});
SubTable = OptogeneticsTable(rows, columns);

varNames= strtrim(cellstr(num2str(trialTimes')));

meltedTable = {};

for iRow = 1: numel(rows)

    testArray = SubTable{iRow,1}{1};
    testTable = array2table(testArray, "VariableNames", varNames);
    testTable{:, 'ROI'} = (1:size(testTable, 1))';
    
    genotypeTable = stack(testTable, {varNames}, ...
        "NewDataVariableName", "Response", "IndexVariableName", "Time");
    
    genotypeTable{:, 'Genotype'} = repmat(SubTable{iRow, 'Genotype'}(1), ...
                                          size(genotypeTable, 1), 1);
    meltedTable{iRow} = genotypeTable;
end

allGenotypesTable = vertcat(meltedTable{:});
tableFileName = 'Optogenetics_L3_C3_Tm1_TimeSeries_Table.csv';
writetable(allGenotypesTable, fullfile(saveFolder, tableFileName))





