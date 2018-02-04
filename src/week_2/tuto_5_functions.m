function funs = tuto_5_functions
  funs.square_function=@square_function;
  funs.squareAndCubeFunction=@squareAndCubeFunction;
  funs.costFunctionJ = @costFunctionJ;
end

function J = costFunctionJ(X, y, theta)
%X is the design matrix containing our training examples
%y is the class labels

m = size(X, 1); % number of training examples
predictions = X * theta; %predictions of hypothesis on all m examples
sqrErrors = (predictions - y) .^ 2; % squared erros

J = 1/ (2 *m) * sum(sqrErrors);
end

function y = square_function(x)
y = x ^ 2;
end

function [y1, y2] = squareAndCubeFunction(x)
y1 = x ^ 2;
y2 = x ^ 3;
end
