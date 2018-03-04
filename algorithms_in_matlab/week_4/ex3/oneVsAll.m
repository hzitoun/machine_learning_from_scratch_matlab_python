function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

for c = 1:num_labels
    %     Set Initial theta
    initial_theta = zeros(n + 1, 1);
    disp('optimizing theta ' + c)
    %
    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50,'Display','off');
    %
    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost
    [theta] = ...
        fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
        initial_theta, options);
    all_theta(c,:) =  theta;
end















% =========================================================================


end
