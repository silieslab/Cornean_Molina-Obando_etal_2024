function outStruct = extractAllParamsFromTable(StimTableCell, varargin) 

switch nargin
    case 1   
        threshold = 0;
        rSquareThreshold = -20;
        alignToOff = false;
        alignToAll = false;
    case 2
        threshold = varargin{1};
        rSquareThreshold = -20;
        alignToOff = false;
        alignToAll = false;
    case 3
        threshold = varargin{1};
        rSquareThreshold = varargin{2};        
        alignToOff = false;
        alignToAll = false;
    case 4
        threshold = varargin{1};
        rSquareThreshold = varargin{2};        
        alignToOff = varargin{3};
        alignToAll = false;
    case 5
        threshold = varargin{1};
        rSquareThreshold = varargin{2};        
        alignToOff = varargin{3};
        alignToAll = varargin{4};
    otherwise
        warning('Too many arguments, cannot process them meaningfully!');
end


[~, ...
 ~, ...
 fitParams, ...
 flyInds, ...
 isAboveThreshold, ...
 tcExt, ...
 tcArgExt, ...
 respInd, ...
 rSquare ] = getAlignedTuningCurveArrayFromTable(StimTableCell, ...
                                                 threshold, ...
                                                 rSquareThreshold, ...
                                                 alignToOff, ...
                                                 alignToAll);

nStims = numel(StimTableCell);
for iStim = 1: nStims
    stimDegreeFactor = 2;
    amplitudeCell{iStim} = fitParams.a1{iStim};
    positionCell{iStim} = stimDegreeFactor * (fitParams.b1{iStim} - 1);
    widthCell{iStim} = stimDegreeFactor  * 2 * sqrt(log(2)) * fitParams.c1{iStim}; % full-width half maximum conversion    amplitudeCell{iStim} = a1;
    nRois(iStim) = sum(isAboveThreshold{iStim});
    nTotalRois(iStim) = numel(isAboveThreshold{iStim}); 
    % Check how many flies have good cells. This neeeds to include rSquare.
    nContributingFlies(iStim) = numel(unique(flyInds{iStim}));
    tcArgExt{iStim} = (tcArgExt{iStim} - 1) * stimDegreeFactor;
end

metadata.nRois = nRois;
metadata.nTotalRois = nTotalRois;
metadata.nContributingFlies = nContributingFlies;

outStruct.metadata = metadata;
outStruct.amplitude = amplitudeCell;
outStruct.position = positionCell;
outStruct.width = widthCell;
outStruct.rSquare = rSquare;
outStruct.respInd = respInd;
outStruct.extremePos = tcArgExt;
outStruct.extreme = tcExt;
