function thresholdedCell = getCellArrayFromIndexThreshold(dataCell, indexCell, threshold) 
    thresholdedCell = cellfun(@(x, y) x(y > threshold), dataCell, indexCell, 'uni', 0);
    