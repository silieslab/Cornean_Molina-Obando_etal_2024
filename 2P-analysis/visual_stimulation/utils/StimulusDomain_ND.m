function [ stimSize, sortedFieldNameInds, fieldNames, stimParamsFlat ] = StimulusDomain_ND( stimParams, fieldNamesToUse )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

if HasBlankEpoch( stimParams )
    stimParams = stimParams(2 : end);
end

stimParamsFlat = FlattenStruct( stimParams );
fieldNames = fieldnames( stimParamsFlat );

%% make intensity field

if any( ismember( fieldNames, 'lum' ) ) && any( ismember( fieldNames, 'contrast' ) )
    for epochInd = 1 : numel( stimParamsFlat )
        stimParamsFlat(epochInd).intensity = stimParamsFlat(epochInd).lum * stimParamsFlat(epochInd).contrast * 2;
    end
    stimParamsFlat = orderfields( stimParamsFlat );
    fieldNames = fieldnames( stimParamsFlat );
end

%% make temporal frequency field

if any( ismember( fieldNames, 'spacing' ) ) && any( ismember( fieldNames, 'stimtrans_mean' ) )
    for epochInd = 1 : numel( stimParamsFlat )
        stimParamsFlat(epochInd).temporalFrequency = stimParamsFlat(epochInd).stimtrans_mean / stimParamsFlat(epochInd).spacing;
    end
    stimParamsFlat = orderfields( stimParamsFlat );
    fieldNames = fieldnames( stimParamsFlat );
end

%% clean up fields that do not vary. not strictly necessary

fieldInd = 1;
while fieldInd <= numel( fieldNames )
    if numel( unique( [stimParamsFlat.(fieldNames{fieldInd})] ) ) == 1
        stimParamsFlat = rmfield( stimParamsFlat, fieldNames{fieldInd} );
        fieldNames = fieldnames( stimParamsFlat );
    else
        fieldInd = fieldInd + 1;
    end 
end

%% select fields using fieldNamesToUse

if ~exist( 'fieldNamesToUse', 'var' ) || isempty( fieldNamesToUse )
    if all( [stimParams.stimtype] == 11 )
        %fieldNamesToUse = {'intensity', 'tau'}; % default stimulus parameters to use
    end
    if all( [stimParams.stimtype] == 46 )
        fieldNamesToUse = {'intensity', 'spacing', 'stimrot_mean', 'temporalFrequency'}; % default stimulus parameters to use
    end
    if all( [stimParams.stimtype] == 50 )
        fieldNamesToUse = {'intensity', 'spacing', 'stimrot_mean', 'stimtrans_mean'}; % default stimulus parameters to use
    end
    if all( [stimParams.stimtype] == 57 )
        fieldNamesToUse = {'stimtrans_amp'}; % default stimulus parameters to use
    end
end

fieldInd = 1;
while fieldInd <= numel( fieldNames )
    if all( ~ismember( fieldNamesToUse, fieldNames{fieldInd} ) )
        stimParamsFlat = rmfield( stimParamsFlat, fieldNames{fieldInd} );
        stimParamsFlat = orderfields( stimParamsFlat );
        fieldNames = fieldnames( stimParamsFlat );
    else
        fieldInd = fieldInd + 1;
    end
end

for fieldInd = 1 : numel( fieldNames )
    axisPeriod(fieldInd) = find( diff( [stimParamsFlat.(fieldNames{fieldInd})] ) ~= 0, 1 );
end
    
[~, sortedFieldNameInds] = sort( axisPeriod );

for fieldInd = 1 : numel( fieldNames )
    stimSize(fieldInd) = numel( unique( [stimParamsFlat.(fieldNames{sortedFieldNameInds(fieldInd)})] ) );
end

end
