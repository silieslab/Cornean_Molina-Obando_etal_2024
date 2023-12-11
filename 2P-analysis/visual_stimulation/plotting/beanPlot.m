function hBeanAx = beanPlot(groupData, groupLabels, colors, offsetScale, hAx, varargin)
    if ~exist('hAx', 'var')   
        hBeanFig = createFullScreenFig;
        hBeanFig.Position(4) = hBeanFig.Position(3) / 2; 
        hAx = gca;
    end
    switch nargin
        case 6
            support = varargin{1};
            plotOrientation = 'horizontal'; % Can change to plotHorizontalBool.
            halveBean = false;
            useAlpha = true;
            bwOption = true;
        case 7
            support = varargin{1};
            plotOrientation = varargin{2};
            halveBean = false;
            useAlpha = true;
            bwOption = true;
        case 8
            support = varargin{1};
            plotOrientation = varargin{2};  
            halveBean = varargin{3};
            useAlpha = true;
            bwOption = true;
        case 9
            support = varargin{1};
            plotOrientation = varargin{2};  
            halveBean = varargin{3};            
            useAlpha = varargin{4};
            bwOption = true;
        case 10
            support = varargin{1};
            plotOrientation = varargin{2};  
            halveBean = varargin{3};            
            useAlpha = varargin{4};
            bwOption = varargin{5};            
        otherwise
            support = 'unbounded';
            plotOrientation = 'horizontal'; % Can change to plotHorizontalBool.
            halveBean = false;
            useAlpha = true;
            bwOption = true;
    end
    hold(hAx, 'on');
    nGroups = numel(groupLabels);
    zeroLineX = [0 0];
    if nGroups == 1
        zeroLineColor = [1 1 1] * 0.5;
        zeroLineY = offsetScale * [1  1.5];
    else
        zeroLineColor = mean(rgb2gray(colors));
        zeroLineY = offsetScale * [1 / 2  (nGroups + 1 / 2)];
    end
    % Flip variables for vertical plotting orientation.
    if strcmp(plotOrientation, 'vertical')
        [zeroLineX, zeroLineY] = deal(zeroLineY, zeroLineX);
    end
    hZeroLine = plot(zeroLineX, zeroLineY, 'LineStyle', '--', 'Color', zeroLineColor);
    bandWidth = nan(1, nGroups);
    for iGroup = 1: nGroups
        data = groupData{iGroup};
        if numel(data) > 1 && range(data) ~=0
            if ~strcmp(support, 'unbounded')
                [bw, f, xi, cdf] = kde(data, 2^10, support(1), support(2));
            else
                [bw, f, xi, cdf] = kde(data, 2^10);
            end
            bandWidth(iGroup) = bw;
        end
    end
    bwMean = nanmedian(bandWidth);

    % 1/2 for the half bean and 2 / 15 for two times the data ticks length.
    if halveBean
        halfOffsetFactor = (1 / 2 + 2 / 15); 
    else
        halfOffsetFactor = 1 + 3 / 15;
    end
    for iGroup = 1: nGroups
        % Define data and plot format.
        data = groupData{iGroup};
        iColor = colors(iGroup, :);
        offset = iGroup * offsetScale * halfOffsetFactor;
        % 1-D strip plot of individual data points.
        dataPointAlpha = 0.1;
        if nGroups == 1
            dataPointColor = [1 1 1] * 0.5;
        else
            dataPointColor = mean(rgb2gray(colors));
        end
        if useAlpha
            dataPointColor = [dataPointColor dataPointAlpha];
        end
        for iDataPoint = 1: numel(data)
            dataPointX = data(iDataPoint) * [1 1];
            if halveBean
                dataPointY = offset + offsetScale / 15 * [-1 0];
            else
                dataPointY = offset + offsetScale / 20 * [-1 1];
            end
            % Flip variables for vertical plotting orientation.
            if strcmp(plotOrientation, 'vertical')
                [dataPointX, dataPointY] = deal(dataPointY, dataPointX);
            end
            plot(dataPointX, dataPointY, 'Color', dataPointColor, 'LineWidth', 0.5);
        end 
        % Get kernel density estimate.
        if bwOption
            bw = bwMean; % Dirty way fo homogeneizing the bandwidth.
            if numel(data) <=10
                bw = numel(data) / 10 * bw;
            end
        else
            bw = bandWidth(iGroup);
        end
%         end
        if numel(data) > 1 && range(data) ~=0
%             if ~strcmp(support, 'unbounded')
%                 [bandwidth, f, xi, cdf] = kde(data, 2^10, support(1), support(2));
                 [f, xi] = ksdensity(data, 'Support', support, 'BoundaryCorrection', 'reflection', 'BandWidth', bw);
%             else
%                 [bandWidth, f, xi, cdf] = kde(data, 2^10);
%             end
            f = f(:)';
            xi = xi(:)';
        else
            f = 1;
            xi = data(1);
        end
%             [f, xi] = ksdensity(data, 'Support', support, 'BoundaryCorrection', 'reflection', 'BandWidth', bw);
        % Normalize the area for plots with different value ranges.
        maxF = max(f);
        f = f / maxF * offsetScale / 2;
        % Half bean.
        if halveBean
            areaXCoords = [xi fliplr(xi)]; % This is the variable value.
            areaYCoords = [f zeros(size(f))] + offset; % This is the variable frequency.
        else
            % Concatenate to make the bean area.
            areaXCoords = [xi  xi(end: -1: 1)]; % This is the variable value.
            areaYCoords = [-f f(end: -1: 1)] + offset; % This is the variable frequency.
        end
        % Flip variables for vertical plotting orientation.
        if strcmp(plotOrientation, 'vertical')
            [areaXCoords, areaYCoords] = deal(areaYCoords, areaXCoords);
        end
        fill(areaXCoords, areaYCoords, iColor, ...
             'FaceAlpha', 0.7, 'EdgeColor', 'none')
        % Now plot quantiles
    %     quantileValue = 0.25;
        lineFormatCell = {'Color', iColor, 'LineWidth', 1};
    %     plotQuantileBean(data, quantileValue, offsetX, lineFormatCell)
    %     quantileValue = 0.5;
    %     lineFormatCell{end} = 2;
    %     plotQuantileBean(data, quantileValue, offsetX, lineFormatCell)
    %     quantileValue = 0.75;
    %     lineFormatCell{end} = 1;
    %     plotQuantileBean(data, quantileValue, offsetX, lineFormatCell)
        lineFormatCell{end} = 2;
        plotBeanMean(data, offsetScale, offset, lineFormatCell, plotOrientation, halveBean)

    end

    hBeanAx = gca;
    prettifyAxes(hBeanAx)
    hBeanAx.XMinorTick = 'off';
    hBeanAx.YMinorTick = 'off';
    % Flip variables for vertical plotting orientation.
    if strcmp(plotOrientation, 'vertical')
        hBeanAx.XTick = offsetScale * (1: nGroups) * halfOffsetFactor;
        hBeanAx.XTickLabel = groupLabels;
        hBeanAx.XTickLabelRotation = 0;
%         if nGroups > 1
%             hBeanAx.PlotBoxAspectRatio = [2 1 1];
%         end
        if halveBean
            hBeanAx.XLim = offsetScale * [(halfOffsetFactor - 1 / 15) ;
                                          ( 1 / 2 + nGroups * halfOffsetFactor + 1 / 15)];
        else
            hBeanAx.XLim = offsetScale * [(- 1 / 2 + halfOffsetFactor - 2 / 15) ...
                                          (+ 1 / 2 + halfOffsetFactor * nGroups + 2 / 15)];
        end
    else
        hBeanAx.YTick = offsetScale * (1: nGroups) * halfOffsetFactor;
        hBeanAx.YTickLabel = groupLabels;
        hBeanAx.YTickLabelRotation = 0;
%         if nGroups > 1
%             hBeanAx.PlotBoxAspectRatio = [1 1 1];
%         end
        % Add option for half bean
        if halveBean
            hBeanAx.YLim = offsetScale * [(halfOffsetFactor - 1.2 / 15) ;
                                          ( 1 / 2 + nGroups * halfOffsetFactor + 1 / 15)];
        else
            hBeanAx.YLim = offsetScale * [(- 1 / 2 + halfOffsetFactor - 2 / 15) ...
                                          (+ 1 / 2 + halfOffsetFactor * nGroups + 2 / 15)];
        end
        if nGroups == 1; axis tight; end
    end
        % Flip variables for vertical plotting orientation.
    if strcmp(plotOrientation, 'vertical')
        hZeroLine.XData = get(gca, 'XLim');
    else
        hZeroLine.YData = get(gca, 'YLim');
    end
end

function plotQuantileBean(data, quantileValue, offset, lineFormatCell)
    [f, xi] = ksdensity(data);
    quantileYCoords = quantile(data, quantileValue) * [1 1];
    [~, quantileInd] = min(abs(quantileYCoords(1) - xi));
    quantileXCoords = [-1 1] * f(quantileInd);
    hold on
    plot(quantileXCoords + offset, quantileYCoords, lineFormatCell{:})
end

function plotBeanMean(data, offsetScale, offset, lineFormatCell, plotOrientation, halveBean)
    [f, xi] = ksdensity(data);
    meanXCoords = nanmean(data) * [1 1];
    if halveBean
        meanYCoords = [0 1] * offsetScale / 2 + offset;
    else
        meanYCoords = [-1 1] * offsetScale / 2 + offset;
    end
    [~, meanInd] = min(abs(meanYCoords(1) - xi));
    if strcmp(plotOrientation, 'vertical')
        [meanXCoords, meanYCoords] = deal(meanYCoords, meanXCoords);
    end
    hold on
    plot(meanXCoords, meanYCoords, lineFormatCell{:})
end