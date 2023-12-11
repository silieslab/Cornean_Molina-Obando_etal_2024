function trialAveragedNormalizedROIs = trialAverageAndNormalizeArray(genotypeStimArray, targetTimes, stimStr)

[nPulses, pulseDurMilliSec, ...
 ISIseconds, nTrials, ...
 ledPowerLEDunits, stimDur, freqHz] = parseStimStrOptogenetics(stimStr);

% I should read the Mark_points.xml, but for now, I know the first stimulus
% comes at t=10s, and the others at t=40.625s,71.875s,102.500s,133.125s,
% for a stimulus that lasts 0.625 s.
stimTimes = 10 + (0:nTrials - 1) .* (ISIseconds + stimDur);
trialStarts = stimTimes - 10;
trialEnds = stimTimes + 20;
    
trialAveragedNormalizedROIs = zeros(size(genotypeStimArray, 2), sum(targetTimes >= trialStarts(1) & targetTimes <= trialEnds(1)));
cellTrials = cell(size(genotypeStimArray, 2), nTrials);
trialLength = inf;
for iCell = 1: size(genotypeStimArray, 2) 
    for iTrial = 1: nTrials
        trialInds = targetTimes >= trialStarts(iTrial) & ...
                    targetTimes <= trialEnds(iTrial);
        cellTrials{iCell, iTrial} = genotypeStimArray(trialInds, iCell);
        trialLength = min(trialLength, sum(trialInds));
    end
    baselines = cellfun(@(x) nanmean(x(floor(0.4* trialLength/3):floor(0.9* trialLength/3)), 1), cellTrials(iCell, :));
    iCellTrials = cellTrials(iCell, :);
    iCellTrialsCropped = cellfun(@(x) x(1:trialLength), iCellTrials, 'UniformOutput', false);
    normalizedROI = bsxfun(@rdivide, bsxfun(@minus, [iCellTrialsCropped{:}], baselines), baselines);
    trialAverage = nanmean(normalizedROI, 2);
    if any(isnan(trialAverage)) %|| max(abs(trialAverage)) > 1
        trialAverage = nan(size(trialAverage));
    end
    if mean(diff(targetTimes)) > stimDur
        pulseTimes = targetTimes(1:length(trialAverage)) >= (stimTimes(1) - mean(diff(targetTimes))) & ...
                     targetTimes(1:length(trialAverage)) <= stimTimes(1) + stimDur + mean(diff(targetTimes));
    else
        pulseTimes = targetTimes(1:length(trialAverage)) >= stimTimes(1) & ...
                     targetTimes(1:length(trialAverage)) <= stimTimes(1) + stimDur;
    end
    trialAverage(pulseTimes) = 0;
    trialAveragedNormalizedROIs(iCell, :) = trialAverage';
end

end