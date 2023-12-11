function [amplitudeArray, positionArray, widthArray, responseIndArray, ...
          categoryLabel] =  getParamsArrayFromStimTable(StimTable, qualityMeasure)

    [fitParams, gof] = getFitParamsStructArrayFromTable(StimTable);
%     domain = StimTable.nonParamTuning{1}(1).domain;

    switch qualityMeasure
        case 'rSquare'
            responseIndArray = [gof.rsquare];
            categoryLabel = 'Goodness of fit (r^2)';
        case 'responseQuality'
            responseIndArray = getRoisFeaturePerChannel(StimTable.responseIndex);
            categoryLabel = 'Response quality index';
        otherwise
            responseIndArray = getRoisFeaturePerChannel(StimTable.responseIndex);
            categoryLabel = 'Response quality index';
    end

    % Rescale fit units to screen units. We just assume the 1 is screen start
    % and number of domain elements is the end. Multiply by 2 deg distance
    % between adjacent bar positions. For the width also multiply by a 2 deg
    % factor
    stimDegreeFactor = 2;
    amplitudeArray = fitParams.a1;
    positionArray = stimDegreeFactor * (fitParams.b1 - 1);
    widthArray = stimDegreeFactor  * 2 * sqrt(log(2)) * fitParams.c1; % full-width half maximum conversion

end