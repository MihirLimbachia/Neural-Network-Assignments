function [centroids, memberships] = kMeans(X, initial_centroids, max_iters)
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

