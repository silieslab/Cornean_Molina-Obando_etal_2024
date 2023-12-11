function [StimSubsetTable, stimRowIndCell] = getStimSubsetTable(PathStimTable, stimSetCellStr, stimInd)
%% This function gets a pathStimTable and a cell of stimuli name strings and a 
% index to the cell of stimuli to retrieve the matching table entries of the
% given stimulus. This also works for extracting cycles of desired stimulus.
stimRowIndCell = cellfun(@(x) strcmp(PathStimTable.stimParamFileName, x), ...
                         stimSetCellStr, 'uni', 0);
StimSubsetTable = PathStimTable(stimRowIndCell{stimInd}, :);

end