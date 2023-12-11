function baseName = FindSessionPrefix(basePath, flyInd, stackInd)
% FINDESSIONPREFIX - Find prefix of data for a particular fly

try 
    search_string = fullfile(basePath, sprintf('*fly%d*', flyInd));
    listing = dir(search_string);
    flyFolder =  fullfile(basePath, listing(1).name);
    search_stack_str = fullfile(flyFolder, '*');
    stackListing = dir(search_stack_str);
    % Delete the '.' and '..' results of dir.
    invalidFileIdx =cell2mat(cellfun(@(x) strcmp(x,'.') || ...
                          strcmp(x,'..') || ...
                          ~isempty(strfind(x, 'SingleImage')) || ...
                          ~isempty(strfind(x, 'ZSeries')) || ...
                          ~isempty(strfind(x, 'BrightnessOverTime')), ...
                          {stackListing.name},'UniformOutput', false));
    stackListing(invalidFileIdx) = [];
    stackListing(~[stackListing.isdir]) = []; % Delete non-directory entries. 2016.04.25
    baseName = fullfile(flyFolder,stackListing(stackInd).name);
%     tokens = regexp(listing(1).name, '^(.+_)[^_]+$','tokens');
%     baseName = fullfile(basePath, tokens{1}{1});
    % disp(sprintf('Found data for fly %d in %s', flyInd, basePath))
catch
    % GRT 2023
    % warning(sprintf('Could not find data for fly %d in %s', flyInd, basePath));
    baseName = '';
end
