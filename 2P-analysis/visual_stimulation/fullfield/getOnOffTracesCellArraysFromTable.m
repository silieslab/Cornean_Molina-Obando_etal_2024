function allNeuronsOnOffTraces = getOnOffTracesCellArraysFromTable(StimTableNeuronStim)

allNeuronsOnOffTraces = cell(1,  size(StimTableNeuronStim, 1));
for iNeuron = 1: size(StimTableNeuronStim, 1)
    allOnOffTracesCell = cat(1, StimTableNeuronStim{iNeuron, 1}.paddedEpochStacks{:});
    allOnOffTracesCell = cellfun(@(x) x', allOnOffTracesCell, 'uni', 0);
    allOffTraces = cat(1, allOnOffTracesCell{:, 1});
    allOnTraces = cat(1, allOnOffTracesCell{:, 2});
    allOnOffTracesArray = [allOffTraces allOnTraces];
    allNeuronsOnOffTraces{iNeuron} = allOnOffTracesArray;
end