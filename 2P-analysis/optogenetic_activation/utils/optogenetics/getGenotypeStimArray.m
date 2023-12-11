function [genotypeStimArray, nFlies, varargout] = getGenotypeStimArray(validTimeSeriesCollection, ...
                                                            genotypeStr, stimStr)

genotypeInds = arrayfun(@(x) ~isempty(strfind(FindSessionPrefix(...
                        x.dataPath{1}, x.dataPath{2}, x.dataPath{3}), ...
                        genotypeStr)), validTimeSeriesCollection);   
genotypeCollection = validTimeSeriesCollection(genotypeInds);
% Check the stimuli
[C, ~, ic] = unique({genotypeCollection.stimulus}');
stimIdx = find(strcmp(C, stimStr));
% genotypeStimCollection = genotypeCollection(ic == stimIdx);
% genotypeInterp = interpSeries(genotypeInds);
% genotypeStimArray = cell2mat(genotypeInterp(ic == stimIdx));
if ~isempty(stimIdx)
    genotypeStimArray = [genotypeCollection(ic == stimIdx).interpData];
    nFlies = numel(unique(arrayfun(@(x) strcat(x.dataPath{1}, ...
                   num2str(x.dataPath{2})), genotypeCollection(ic == stimIdx), ...
                   'UniformOutput', false)));
    varargout{1} = genotypeInds;
    tmpInds = find(genotypeInds);
    varargout{2} = tmpInds(ic == stimIdx);
else
    genotypeStimArray = [];
    nFlies = 0;
    varargout{1} = [];
    varargout{2} = [];
end

end
