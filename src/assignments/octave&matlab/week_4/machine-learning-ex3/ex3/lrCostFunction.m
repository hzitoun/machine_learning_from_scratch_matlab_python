function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
prediction = sigmoid(X * theta);

cost_y_1 = (1 - y) .* log(1 - prediction);
cost_y_0 = -1 .* y .* log(prediction);

grad_dim = size(X,2);

J = 1/m * sum(cost_y_0 - cost_y_1) + (lambda/(2 * m)) * sum(theta(2:grad_dim) .* theta(2:grad_dim));

grad_without_regul = 1/m .* X' * (prediction - y) ;

grad = [grad_without_regul(1); grad_without_regul(2:grad_dim) + (lambda/m) .* theta(2:grad_dim)];

end
