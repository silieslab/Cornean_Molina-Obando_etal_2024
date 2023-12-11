function setFontForThesis(hAx, hFig)
fontToUse = 'Helvetica';
set(hAx,'FontSize', 8)
set(findall(hFig,'type','text'),'FontSize',8)
arrayfun(@(x) set(get(x, 'Title'), 'FontSize', 10, 'FontWeight', 'normal'), hAx);
set(hAx,'FontName', fontToUse)
set(findall(hFig,'type','text'),'FontName', fontToUse)
end