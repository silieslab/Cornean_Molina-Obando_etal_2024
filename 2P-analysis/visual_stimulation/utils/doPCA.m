function [U, S, V, variances, pcaCoords, dataReduced] = doPCA(data)
% This uses other function than the matlab default. Data has samples in
% rows and variables in columns.
% For reproducible results.
rand('seed', 4);
% PCA of centered data. Here 20 components only.
[U, S, V] = pca(data - mean(data), 20);
% Calculate explained variances.
variances = diag(S).^2 / (size(U, 1) - 1);
variances = variances / sum(var(data)) * 100;
% Coefficients in PCs.
pcaCoords = U * S;
% Reconstructed data in reduced dimensions.
dataReduced = U * S * V';

end
