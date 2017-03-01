function memberships = findClosestCentroids(X, centroids)
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
