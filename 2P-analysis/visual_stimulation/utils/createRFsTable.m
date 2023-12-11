function DataTable = createRFsTable(rfArray, stimParamString)
    degPerBar = 2;
    relativePosition = degPerBar * (1 :  size(rfArray, 2));
    dataCell = mat2cell([relativePosition; rfArray], ones(1, size(rfArray, 1) + 1), ones(1, size(rfArray, 2)));
    labelColumn = [stimParamString; sprintfc('ROI%d', (1: size(rfArray, 1))')];
    labelDataCell = [labelColumn dataCell]';
    DataTable = cell2table(labelDataCell(2: end, :), 'VariableNames', labelDataCell(1, :));
end

% function exportRFsTableToExcel(rfArray)
% 
% end
