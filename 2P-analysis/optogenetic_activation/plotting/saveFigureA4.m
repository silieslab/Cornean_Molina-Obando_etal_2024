function saveFigureA4(hFigure, folderName, figureFileName, dpiResolution, fixLine, orientation)
% saveFigureA4Landscape(hFigure, folderName, figureFileName, dpiResolution, fixLine, orientation)
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
    switch orientation
        case 'Portrait'
            orientation = 'Landscape';
        case 'Landscape'
            orientation = 'Portrait';
    end
    set(0, 'CurrentFigure', hFigure);
    set(gcf, 'PaperType', 'A4', 'PaperPositionMode', 'auto', ...
        'PaperOrientation', orientation);
    set(gcf,'PaperSize', fliplr(get(gcf,'PaperSize')))
    set(gcf, 'PaperPositionMode', 'auto', 'Renderer', 'painters');
    if fixLine
        print(gcf, '-depsc2', ['-r' num2str(dpiResolution)], ...
              [folderName strcat(figureFileName)]);
    end
    set(0, 'CurrentFigure', hFigure);
    print(gcf, '-fillpage', '-dpdf', ['-r' num2str(2 * dpiResolution)], ...
          [folderName strcat(figureFileName)]);
    if fixLine
        fixPSlinestyle([folderName strcat(figureFileName, '.eps')], [folderName strcat(figureFileName, 'Fix.eps')]);
    end
    close(hFigure);
end
