function [ figHandle, colorFlag ] = PlotImagingStruct( imagingStruct, colorFlag )
% PLOTIMAGINGSTRUCT takes either an up to 3 channel image or an
% imagingStruct and plots it appropriately

if isstruct( imagingStruct ) % if imagingStruct is a true imagingStruct
    I = imagingStruct.stackRaw;
    I = nanmean( I, 3 );
    filePath = regexp(imagingStruct.fullPath, [imagingStruct.flyDir '(.*)'], ...
                      'tokens', 'once');
    hText(2) = text( 0, 4, [imagingStruct.flyDir filePath{1}] );
    titleString = {[imagingStruct.experimentParentDir ...
                    imagingStruct.experimentDir filesep], ...
                   [imagingStruct.flyDir filePath{1}], ...
                   ['stackInd=' num2str( imagingStruct.stackInd ) ...
                    ', nFrames=' num2str( imagingStruct.nFrames ), ...
                    ', isAligned=' num2str( imagingStruct.isAligned ) ...
                    ', isAlignedAbs=' num2str( imagingStruct.isAlignedAbs )  ...
                    ', isTrialAveraged=' num2str( imagingStruct.isTrialAveraged )]};
    
    if ~exist( 'colorFlag', 'var') || isempty( colorFlag )
        colorFlag = 'rgb';
    end
elseif isnumeric( imagingStruct ) % if imagingStruct is an image or a map
    I = imagingStruct;
    titleString = 'not an imagingStruct - no metadata';
    
    if ~exist( 'colorFlag', 'var') || isempty( colorFlag )
        switch size( I, 3 ) % one- and two-channel data are considered rgb images by default. three-channel data that are appropriately scaled are considered hsv maps.
            case 1
                colorFlag = 'rgb';
            case 2
                colorFlag = 'rgb';
            case 3
                if all( max( max( I ) ) <= 1 ) && all( min( min( I ) ) >= 0 ) && any( I(:) ~= 0 ) % if all channels are scaled to the unit interval, assume hsv. otherwise the data is scaled to native units. then assume rgb.
                    colorFlag = 'hsv';
                else
                    colorFlag = 'rgb';
                end
            otherwise
                colorFlag = 'rgb';
        end
    end
else
    disp( 'No base image to display.' );
    disp( 'WARNING: Return arguments are empty.' );

    figHandle = [];
    
    return;
end

[nRows, nCols, nChannels] = size( I );

figHandle = gcf;

switch colorFlag
    case 'rgb'
        IScaled = zeros( [nRows, nCols, 3] );
        for chanInd = 1 : nChannels
            IScaled(:, :, chanInd) = I(:, :, chanInd) / max( max( I(:, :, chanInd) ) ) * 2; % saturate by multipling by 2
        end
        if nChannels == 1
            IScaled(:, :, 2) = IScaled(:, :, 1);
            IScaled(:, :, 3) = IScaled(:, :, 1);
        end
        imshow( IScaled );
    case 'hsv'
        if nChannels < 2
            I(:, :, 2) = ones( nRows, nCols );
        end
        if nChannels < 3
            I(:, :, 3) = ones( nRows, nCols );
        end
        image( hsv2rgb( I ) );
        axis image;
        axis ij;
end

title( titleString );

end
