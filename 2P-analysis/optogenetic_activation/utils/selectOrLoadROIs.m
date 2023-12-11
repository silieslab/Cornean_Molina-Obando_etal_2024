function [ roiStruct ] = selectOrLoadROIs( basePath, flyInd, stackInd, ...
                                           imagingStruct, uiSelection, varargin )
% selectOrLoadROIs Load existent roiStruct from file or select ROIs (do not
% choose overlapping regions!)
% 
% Inputs
% ------
% 
% basePath:         the path up to the experiment folder.
% flyInd:           the number of fly on that experiment.
% stackInd:         index of stimulus recorded for the given fly by flyInd.
% imagingStruct:    data structure containing the images and corresponding
%                   metadata from where to select the ROIs.
% uiSelection:      boolean to choose between manual selection(true) or
%                   automatic file loading (false)
% varargin:         variables for slection of data with one time series per
%                   layer with multiple channels and several cycles per 
%                   time series. First stimulus number, then channel.
% Outputs
% -------
% 
% roiStruct:        structure containing the mask of the selected ROIs.
% 
    switch nargin
        case 6
            cycleInd = varargin{1};
            filenameSuffix = ['Cycle_' num2str(cycleInd)];
        case 7
            cycleInd = varargin{1};
            channelInd = varargin{2};
            filenameSuffix = ['Cycle_' num2str(cycleInd) ...
                              '-Channel_' num2str(channelInd)];
        otherwise
            filenameSuffix = '';
    end
    
    baseName = FindSessionPrefix(basePath, flyInd, stackInd);
    FilterSpec = regexprep(baseName, ['Mat' filesep], ['ROI' filesep]);
    if isdir(FilterSpec)
        if uiSelection
            [FileName, PathName, ~] = uigetfile([FilterSpec filesep '*' filenameSuffix '.mat']);
        else
            PathName = FilterSpec;
            try
                FileName = dir([FilterSpec filesep '*' filenameSuffix '.mat']);
                FileName = FileName.name;
            catch
                % Could not find any file matching.
                FileName = '';
            end
        end
        try
            outputStruct = open(fullfile(PathName, FileName));
            if isfield(outputStruct, 'roiStruct')
                roiStruct = outputStruct.roiStruct;
            end
        catch
            warning(['Selected file does not contain ROIs. '...
                     'Please select new non-overlaping ROIs'])
            [ ~, roiStruct ] = SelectROIs_Freehand( imagingStruct, filenameSuffix );
        end
    else
        [ ~, roiStruct ] = SelectROIs_Freehand( imagingStruct, filenameSuffix );
    end

end

