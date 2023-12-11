function [nPulses, pulseDurMilliSec, ISIseconds, nTrials, ledPowerLEDunits, stimDur, varargout] = parseStimStrOptogenetics(stimStr)
    splitStrCell = regexp(stimStr, '_','split');
    if numel(splitStrCell) == 5
        [pulse, pulseDur, ISI, repeats, ledPower] = deal(splitStrCell{:});
        if nargout > 6; varargout{1} = []; end
    elseif numel(splitStrCell) == 6
        [pulse, pulseDur, freq, ISI, repeats, ledPower] = deal(splitStrCell{:});
        freqHz = getNumberFromStr(freq);
        if nargout > 6; varargout{1} = freqHz; end
    end

        function number = getNumberFromStr(str)
            [token, ~] = regexp(str, '(\d+).*', 'tokens', 'match');
            number = str2double(token{:}{:});
        end

    nPulses = getNumberFromStr(pulse);
    pulseDurMilliSec = getNumberFromStr(pulseDur);
    ISIseconds = getNumberFromStr(ISI);
    nTrials = getNumberFromStr(repeats);
    ledPowerLEDunits = getNumberFromStr(ledPower);
    if exist('freqHz', 'var')
        stimDur = nPulses / freqHz;
    else 
        stimDur = pulseDurMilliSec / 1000;
    end
end