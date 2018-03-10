function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

values = [0.01,0.03,0.1,0.3,1,3,10,30];

max_error = 1;

for i=1:8
    for j=1:8
        current_c = values(i);
        current_sigma = values(j);
        
        %ALWAYS train the model on training sets (X and y)
        model= svmTrain(X, y, current_c, @(x1, x2) gaussianKernel(x1, x2, current_sigma)); 
        
        %AND evaluate it on cross validation set
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
           
        if error < max_error
            max_error = error;
            C = current_c;
            sigma = current_sigma;
        end
        
    end
end





% =========================================================================

end
