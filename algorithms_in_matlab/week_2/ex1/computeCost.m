function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
predictions = X * theta; %predictions of hypothesis on all m examples
sqrErrors = (predictions - y) .^ 2; % squared erros
J = 1/ (2 *m) * sum(sqrErrors);


% =========================================================================

end
