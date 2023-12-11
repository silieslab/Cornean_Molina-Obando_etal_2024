function featureArray = getRoisFeaturePerChannel(featureCellArray)
for iChannel = 1: size(featureCellArray{1}, 1)
    featureArrayTemp = cellfun(@(x) x(iChannel, :), featureCellArray, 'uni', 0);
    featureArray(iChannel, :) = [featureArrayTemp{:}];
end
end