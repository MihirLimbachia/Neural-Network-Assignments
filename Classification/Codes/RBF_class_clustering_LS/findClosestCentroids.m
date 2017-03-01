function memberships = findClosestCentroids(X, centroids)
%   Parameters
%     X         - The data set, with one sample per row.
%     centroids - The current centroids, one per row.
%   
%   Returns
%     A column vector containing the index of the closest centroid (a value
%     between 1 - k) for each corresponding data point in X.
k = size(centroids, 1);
m = size(X, 1);

memberships = zeros(m, 1);

distances = zeros(m, k);

for i = 1 : k
    diffs = bsxfun(@minus, X, centroids(i, :));
    
    sqrdDiffs = diffs .^ 2;
    distances(:, i) = sum(sqrdDiffs, 2);

end

[minVals memberships] = min(distances, [], 2);

end