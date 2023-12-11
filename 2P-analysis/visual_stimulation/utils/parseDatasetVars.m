function [saveFolder, figDir, stimSetCellStr] = parseDatasetVars(datasetInd)

switch datasetInd
    case 1
        saveFolder = 'D:\Data_LGRTL\Tm9GCaMP6f_2016-2017\';
        figDir = 'P:\Documents\Figures\Paper\Tm9\fffOld\';
        stimSetCellStr = getStimSetCellStr(1);
    case 2
        saveFolder = 'D:\Data_LGRTL\Tm9GCaMP6f_2018-2019\';
        figDir = 'P:\Documents\Figures\Paper\Tm9\fffNew\';
        stimSetCellStr = getStimSetCellStr(2);
    case 3
        saveFolder = 'D:\Data_LGRTL\Tm9GCaMP6f_2018-2019\';
        figDir = 'P:\Documents\Figures\Paper\Tm9\noise\';
        stimSetCellStr = getStimSetCellStr(3);
end