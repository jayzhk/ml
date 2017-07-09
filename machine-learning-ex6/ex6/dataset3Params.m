function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
m_C = size(C_array, 2);
m_Sigma = size(sigma_array ,2);
comb = zeros(m_C * m_Sigma, 2);

for i = 1 : m_C
    for j = 1 : m_Sigma;
        comb(m_C *(i-1) + j,:) = [C_array(i), sigma_array(j)];
    end
end

comb = [comb zeros(size(comb, 1), 1)];

for i = 1: size(comb,1)
    
	model= svmTrain(X, y, comb(i,1), @(x1, x2) gaussianKernel(x1, x2, comb(i, 2)));
	predictions = svmPredict(model, Xval);
	comb(i, 3) = mean(double(predictions ~= yval));
end


[V, I] = min(comb(:, 3));

C = comb(I,1)
sigma = comb(I, 2)






% =========================================================================

end
