function [ roiTcs ] = ApplyROIsToStack_TimeCourse( I, roiStruct )
% takes a 3D stack and roiStruct and extracts time courses from the stack
% for the ROIs. adapted from matlab file exchange 26482. this approach is
% ~3x faster than my original approach, which obtained a ROI's timecourse
% by repmat'ing the mask to the size of the stack, then multiplying, and
% then finding mean intensity manually, rather than with regionprops. this
% approach is commented below:
% 
% tcChan = squeeze( I(:, :, :, channel) );
% 
% roiTcRaw = zeros( roiStruct.nROIs, size( tcChan, 3 ) );
% for roiInd = 1 : roiStruct.nROIs
%     % calculate the raw ROI time-course
%     roiBwMask = repmat( roiStruct.bwMaskStack(:, :, roiInd), [1 1 size( tcChan, 3 )] );
%     roiArea = regionprops( roiStruct.bwMaskStack(:, :, roiInd), 'Area' );
%     roiArea = roiArea.Area;
%     roiTcRaw( roiInd, : ) = squeeze( sum( sum( tcChan .* roiBwMask, 1 ), 2 ) ) / roiArea;
% end

tic;

nFrames=size( I, 3 );

roiTcs=zeros( [roiStruct.nROIs, nFrames] );   %preallocate memory for poparray

parfor frameInd = 1 : nFrames
    frame = I( :, :, frameInd );
    frameROIs = regionprops(roiStruct.bwLabel, frame, 'MeanIntensity');
    frameROIs = {frameROIs.MeanIntensity};
    roiTcs(:,frameInd) = cell2mat( frameROIs );
end

toc;

end
