function [fitParams, gof] = getFitParamsStructArrayFromTable(StimTable)
nChannels = size(StimTable.responseIndex{1}, 1); % (Dirty?) wa to get nChannels.
fitCellArray = StimTable.paramTuning;
fitStructArray = [fitCellArray{:}];
fieldNames = fieldnames(fitStructArray(1).fit);
gof = FlattenStruct(reshape([fitStructArray.gof], nChannels, []));


for iField = 1: numel(fieldNames)
     fitParams.(fieldNames{iField}) = arrayfun(@(x) x.fit.(fieldNames{iField}), fitStructArray);
end

end
