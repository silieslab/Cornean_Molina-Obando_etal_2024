function [ bwMaskStack, roiStruct ] = SelectROIs_Freehand( imagingStruct, fileNameSuffix )
% takes an input image I and returns a ROIstruct out. the ROIs
% are chosen by freehand user input using the roipoly function. this is for
% manual selection

figure;
figHandle = PlotImagingStruct( imagingStruct );
nRows = diff( get( get( figHandle,'CurrentAxes' ), 'YLim' ) );
nCols = diff( get( get( figHandle,'CurrentAxes' ), 'XLim' ) );

disp( 'Press any key to return.' );

isCompleteLogical = false;
bwMaskInd = 1;
bwMask = zeros( [nRows, nCols] );
bwLabel = zeros( [nRows, nCols] );
bwMaskStack = zeros( nRows, nCols, 1 );
while ~isCompleteLogical
    tmpFrame = roipoly;
    bwMask = max( bwMask, tmpFrame );
    bwLabel = max( bwLabel, bwMaskInd * tmpFrame );
    bwMaskStack(:, :, bwMaskInd) = tmpFrame;
    
    alphamask( bwMaskStack(:, :, bwMaskInd ), [1 0 1], 0.33 );
    
    bwMaskInd = bwMaskInd + 1;
    isCompleteLogical = waitforbuttonpress;
end

close;

roiStruct.bwMask = bwMask;
roiStruct.bwLabel = bwLabel;
roiStruct.bwMaskStack = bwMaskStack;
roiStruct.nROIs = bwMaskInd - 1;
roiStruct.roiType = 'Freehand';
roiStruct.uniqueID = datestr( now, 30 );

if isstruct( imagingStruct )
    % add fileName and stackInd
    roiStruct.fileName = imagingStruct(1).fileName;
    roiStruct.stackInd = imagingStruct(1).stackInd;
    
    % set up file paths and names
    % Modified save path. Luis 17.03.2016.
    roiFileDir = [regexprep( imagingStruct(1).experimentParentDir, ['([^\' filesep ']+)\' filesep '$'], ['$1ROI\' filesep] ) ...
                  imagingStruct(1).experimentDir filesep];
    roiFileDir = [roiFileDir imagingStruct(1).flyDir filesep ...
                  imagingStruct(1).timeSeriesDir filesep];
    
    
    if ~isdir( roiFileDir ) mkdir( roiFileDir ); end
    
    if ~exist( 'fileNameSuffix', 'var' ) || isempty( fileNameSuffix )
        fileNameSuffix = '';
        roiStruct.fileNameSuffix = fileNameSuffix;
    else
        roiStruct.fileNameSuffix = fileNameSuffix;
        fileNameSuffix = ['-' fileNameSuffix];
    end
    
    roiStructFileNameRoot = ['roiStruct-' roiStruct.uniqueID '-' roiStruct.roiType fileNameSuffix];
    save( [roiFileDir roiStructFileNameRoot '.mat'], 'roiStruct', '-v7.3', '-mat' );
end

end
