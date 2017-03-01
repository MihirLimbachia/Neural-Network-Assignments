function [centroids, memberships] = kMeans(X, initial_centroids, max_iters)
%   Parameters
%     X                 - The dataset, with one example per row.
%     initial_centroids - The initial centroids to use, one per row (there
%                         should be 'k' rows).
%     max_iters         - The maximum number of iterations to run (k-means will
%                         stop sooner if it converges).
%   Returns
%     centroids    -  A k x n matrix of centroids, where n is the number of 
%                    dimensions in the data points in X.
%     memberships  - A column vector containing the index of the assigned 
%                    cluster (a value between 1 - k) for each corresponding 
%                    data point in X.
k = size(initial_centroids, 1);

centroids = initial_centroids;
prevCentroids = centroids;

for (i = 1 : max_iters)
    
    memberships = findClosestCentroids(X, centroids);
    centroids = computeCentroids(X, centroids, memberships, k);
    if (prevCentroids == centroids)
        break;
    end
    prevCentroids = centroids;
end

end

