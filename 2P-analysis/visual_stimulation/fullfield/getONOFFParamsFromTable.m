function out = getONOFFParamsFromTable(StimTableNeuronStim, varargin)

if nargin > 1
    nToExcludeFromEnd = varargin{1};
else
    nToExcludeFromEnd = 1;
end

for iNeuron = 1: size(StimTableNeuronStim, 1)
    responseInd{iNeuron} = [StimTableNeuronStim{iNeuron, 1}.responseIndex{:}];
    params = [StimTableNeuronStim{iNeuron, 1}.paramTuning{:}];
    polarityWithoutBG = arrayfun(@(x) x.polarityIndex(1: end - nToExcludeFromEnd), params, 'uni', 0);
    onWithoutBG = arrayfun(@(x) x.onMean(1: end - nToExcludeFromEnd), params, 'uni', 0);
    offWithoutBG = arrayfun(@(x) x.offMean(1: end - nToExcludeFromEnd), params, 'uni', 0);
    polarity{iNeuron} = [polarityWithoutBG{:}];
    onMean{iNeuron} = [onWithoutBG{:}];
    offMean{iNeuron} = [offWithoutBG{:}];
    lateToExtremeRatioOFF{iNeuron} = [params.lateToExtremeRatioOFF];
    halfRiseOFF{iNeuron} = ([params.halfRiseOFF] - 1) / 10; % Assumes interpolating at 10 Hz.
    timeToPeakOff{iNeuron} = ([params.argExtremeOFF] - 1) / 10; % Assumes interpolating at 10 Hz.
end

out.polarity = polarity;
out.onMean = onMean;
out.offMean = offMean;
out.lateToExtremeRatioOFF = lateToExtremeRatioOFF;
out.halfRiseOFF = halfRiseOFF;
out.timeToPeakOff = timeToPeakOff;
out.responseInd = responseInd;
end
