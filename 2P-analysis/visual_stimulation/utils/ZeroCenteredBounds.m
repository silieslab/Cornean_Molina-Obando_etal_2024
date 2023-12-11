function [ bounds ] = ZeroCenteredBounds( data, clipPrctileRange )
% ZEROCENTEREDINTERVAL returns a zero centered interval that surrounds
% input data. useful for determining the color interval for imagesc map
% plotting. to set minVal and maxVal to true min and max, use
% clipPrctileRange = [0 100].

minVal = prctile( data(:), clipPrctileRange(1) );
maxVal = prctile( data(:), clipPrctileRange(2) );

if isnan( minVal ) || isnan( maxVal )
    bounds = [-1 1];
else
    if sign( minVal ) == sign( maxVal )
        if sign( minVal ) < 0
            bounds = [minVal -minVal];
        elseif sign( minVal ) > 0
            bounds = [-maxVal maxVal];
        else
            bounds = [-1 1];
        end
    else
        scaleVal = max( abs( minVal ), abs( maxVal ) );
        bounds = [-scaleVal scaleVal];
    end
end

end
