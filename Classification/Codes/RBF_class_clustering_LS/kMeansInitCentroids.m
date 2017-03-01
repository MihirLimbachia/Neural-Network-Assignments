function centroids = kMeansInitCentroids(X, k)
%   Parameters
%     X  - The dataset, one data point per row.
%     k  - The number of cluster centers.
%
%   Returns
%     A matrix of centroids with k rows.
centroids = zeros(k, size(X, 2));

randidx = randperm(size(X, 1));

centroids = X(randidx(1:k), :);

end

