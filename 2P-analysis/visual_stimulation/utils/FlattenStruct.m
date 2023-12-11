function [ structOut, cellArray, fieldNames ] = FlattenStruct( structIn, fieldPrefix )
% FLATTENSTRUCT flattens a hierarchical structure array into a flat version.
% field names are concated with _

cellArray = struct2cell( structIn );
fieldNames = fieldnames( structIn );

fieldInd = 1;
while fieldInd <= numel( fieldNames )
    if exist( 'fieldPrefix', 'var' ) && ~isempty( fieldPrefix )
        fieldNames{fieldInd} = [fieldPrefix '_' fieldNames{fieldInd}];
    end
    if isstruct( cellArray{fieldInd, 1, 1} )
        subStruct = [cellArray{fieldInd, :, :}];
        [~, subCellArray, subFieldNames] = FlattenStruct( subStruct, fieldNames{fieldInd} );
        
        cellArray = [cellArray; subCellArray];
        fieldNames = [fieldNames; subFieldNames];
        
        cellArray(fieldInd, :, :) = '';
        fieldNames(fieldInd) = '';
    else
        fieldInd = fieldInd + 1;
    end
end

structOut = cell2struct( cellArray, fieldNames );
structOut = orderfields( structOut );

end
