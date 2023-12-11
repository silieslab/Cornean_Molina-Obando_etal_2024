% Set colormap to green purple.
function colormapG2P = setFavoriteColormap
%% Set the green purple divergent map from cpt-city maps.
cptFile =  'greenPurple'; %'PRGn_11'; %
colormapG2P = cptcmap(cptFile, 'ncol', 256);
colormap(gca, colormapG2P);

end