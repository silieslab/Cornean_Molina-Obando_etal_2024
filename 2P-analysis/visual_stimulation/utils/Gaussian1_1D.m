function y = Gaussian1_1D(A, xdata)

% Gaussian function
% A: parameters of the Gaussian function
%       A(1) = a1, A(2) = mu1, A(3) = sigma1, A(4) = b
%       y = a1 * exp( -((x - mu1) ^ 2) / (2 * sigma1 ^ 2 ) ) + b
% xdata: a vector which contains x values

y = A(1) * exp( -((xdata - A(2)) .^ 2) / (2 * A(3) ^ 2) ) + A(4);
