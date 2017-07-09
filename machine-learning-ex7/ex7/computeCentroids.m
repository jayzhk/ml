function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for i = 1 : K
    % 1. idx == i find all the rows that belongs to centroid i
    % 2. (idx == i) * ones(1, n) expand the class to all the columns
    % 3. column wise multiplication to get all values of row and sum it
    % 4. divide by the number of instances belows to the same class
    % 5. assign to the correspoinding centroid
    % centroids(i,:) = sum(X .* ((idx == i) * ones(1, n)))/sum(idx == i);
    centroids(i,:) = sum(X .* (repmat((idx == i), [1, n])))/sum(idx == i);
end



    







% =============================================================


end

