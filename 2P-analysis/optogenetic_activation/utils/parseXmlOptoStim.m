function name = parseXmlOptoStim(filename)

if ~exist(filename, 'file'), error(['Could not find file: ' filename]); end

%%
% Import the XPath classes
import javax.xml.xpath.*

% Construct the DOM.
% filename = which('demos/demos.xml');
try
    tic;
    experimentDoc = xmlread(filename);
    elapsedTimeXmlRead = toc;
    fprintf('Time to create XML tree:%f s\n',elapsedTimeXmlRead);
catch
    error('Failed to read XML file %s.',filename);
end

% Create an XPath expression.
factory = XPathFactory.newInstance;
xpath = factory.newXPath;

%% Extract current information.
% Navigate until the Name attribute of the root node PVMarkPointSeriesElements.
try
    expression = xpath.compile('/PVMarkPointSeriesElements/@Name');
    nodeList = expression.evaluate(experimentDoc,XPathConstants.NODESET);
    % Extract the value into the Experiment struct.
    name = char(nodeList.item(0).getNodeValue.toString());
catch
    warning('Old stimulation protocol, skipping this recording.');
    name = 0;
end

end