function [tcArray, responseIndArray, rSquareArray, fitParams, varargout] = getTuningCurveArrayFromTable(StimTable)
%% Show some raw data.
% Get the non-parametric tuning struct to show the tuning curves. Here
% somehow some flies saw stimuli that had more or less bar positions, but 
% those flies had no good cells (above 0.5 response quality index).
% The dimensions of the tuningStruct is 1 x nChannels, each element is then
% 1 x nTuningMethods, here I use the roi function to get all structs for 
% one method into an array.
nonParamTunings = getRoisFeaturePerChannel(cellfun(@(x) x', StimTable.nonParamTuning, 'uni', 0));
if isfield(nonParamTunings(1), 'tuningMethods')
    tuningMethods = nonParamTunings(1).tuningMethods;
    tuningMethodInd = strcmp(tuningMethods, 'extreme');
    tcCellArray = arrayfun(@(x) x.tc{tuningMethodInd}, nonParamTunings, 'uni', 0);
    tcExtCellArray = arrayfun(@(x) x.tcParams{tuningMethodInd}.extremeResp, nonParamTunings, 'uni', 0);
    tcArgExtCellArray = arrayfun(@(x) x.tcParams{tuningMethodInd}.argExtreme, nonParamTunings, 'uni', 0);

    domainCellArray = arrayfun(@(x) x.domain, nonParamTunings, 'uni', 0);
    domainSize = cellfun(@numel, domainCellArray);
    tcCellArray = filterBgInvalidRois(tcCellArray);
    tcExtCellArray = filterBgInvalidRois(tcExtCellArray);
    tcArgExtCellArray = filterBgInvalidRois(tcArgExtCellArray);
    
    % Do the same for responseIndex.
    responseIndArray = getRoisFeaturePerChannel(StimTable.responseIndex(domainSize == mode(domainSize)))';

    % Now that all is homogeneous we can concatenate the arrays.
    tcArray = cell2mat(cellfun(@squeeze, tcCellArray, 'uni', 0)');
    tcExtArray = cell2mat(cellfun(@(x) squeeze(x), tcExtCellArray, 'uni', 0));
    tcArgExtArray = cell2mat(cellfun(@(x) squeeze(x), tcArgExtCellArray, 'uni', 0));
    % Alternative way.
    % tcCellArray = [cellfun(@squeeze, tcCellArray, 'uni', 0)'];
    % cat(1, tcCellArray{:});
    % Lets keep plotting. 
    % Make sure all ROIs have a tuning curve and a quality
    % index.
    assert(numel(responseIndArray) == size(tcArray, 1) && ...
           numel(responseIndArray) == numel(tcExtArray) && ...
           numel(responseIndArray) == numel(tcArgExtArray), ...
           ['Mismatch of number of ROIs for tuning curve array and response' ...
            'quality index array.']);
    varargout{1} = tcExtArray;
    varargout{2} = tcArgExtArray;
    
else
    domainSize = ones(1, numel(StimTable.responseIndex));
    % Do the same for responseIndex.
    responseIndArray = getRoisFeaturePerChannel(StimTable.responseIndex(domainSize == mode(domainSize)))';

    tcArray = [];
    varargout{1} = [];
    varargout{2} = [];
end

% We can load the rSquare of the Gaussian fit.
[fitParams, gof] = getFitParamsStructArrayFromTable(StimTable(domainSize == mode(domainSize), :));
rSquareArray = [gof.rsquare];

    function outputCellArray = filterBgInvalidRois(inputCellArray)
        
    % Actually the last tuning curve is from the background. Check it now!
    bgRois = cellfun(@(x) x.backgroundRois,  StimTable.roiMeta);
    invalidRois = cellfun(@(x) x.invalidRois,  StimTable.roiMeta, 'uni', 0);
    % Lets get rid of background ROIs. Later check if invalidRois give trouble.
    for iFly = 1: numel(bgRois)
        nRois = size(inputCellArray{iFly}, 2);
        roisToKeep = setdiff(1: nRois, [bgRois(iFly) invalidRois{iFly}]);
        inputCellArray{iFly} = inputCellArray{iFly}(:, roisToKeep, :);
    end
    % Checkn that the stimuli are same size and discard the ones that are
    % different (outliers). Anyways they did not contribute good cells for this
    % stimulus in particular.
    outputCellArray = inputCellArray(domainSize == mode(domainSize));    
    
    end


end