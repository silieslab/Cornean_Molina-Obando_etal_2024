function [ blankBool ] = HasBlankEpoch( dataStruct )
% if dataStruct is a stimParams struct, evaluates whether the stimParams
% struct contains a blank bout. there is a blank bout by definition when
% randomize is set to 1. there is also a blank bout when the first
% condition specified is a blank.
%
% if dataStruct is a timingStruct struct, returns hasBlankEpoch field if
% available. otherwise evaluates whether the timingStruct is for a stimulus
% parameter file that included a blank condition. evaluates to true if the
% timingStruct is the output of TrialAverageAll, which create a blank epoch.
% evaluates to true if every other bout is epoch 0, which can occur when
% randomize was set to 1, i.e. cycling between conditions and the blank.
% there can be false positives using this last condition (2 bouts only of
% randomize = 0, insufficiently many bouts of randomize = 2).

if isfield( dataStruct, 'randomize' ) % if dataStruct is a stimParams struct
    blankBool = all( [dataStruct.randomize] == 1 ) || ...
        (dataStruct(1).stimtype == 11 && ...
        dataStruct(1).lum == 0.5 && ...
        dataStruct(1).contrast == 0 && ...
        dataStruct(1).spacing == 120 && ...
        dataStruct(1).duration == dataStruct(1).tau);
elseif isfield( dataStruct, 'hasBlankEpoch' ) && isfield( dataStruct, 'stimParamFileName' )% if dataStruct is a timingStruct struct
    exceptionStimuli = {'FullField_ONOFF_1.0_2s_BG_0.5_4s_Weber_NonRand.txt'};
    if strcmp(dataStruct.stimParamFileName, exceptionStimuli)
        blankBool = 1;
    else
        blankBool = dataStruct.hasBlankEpoch;
    end
elseif isfield( dataStruct, 'stimulusOutputRaw' ) % if dataStruct is a timingStruct struct but lacks hasBlankEpoch field
    blankBool = (isfield( dataStruct, 'isTrialAveraged' ) && dataStruct.isTrialAveraged) || ...
        all( dataStruct.boutEpochInds(1 : 2 : end) == 0 );
end
    
end
