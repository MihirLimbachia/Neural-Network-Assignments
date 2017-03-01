function centroids = computeCentroids(X, prev_centroids, memberships, k)
[m n] = size(X);

centroids = zeros(k, n);

for (i = 1 : k)
    if (~any(memberships == i))
        centroids(i, :) = prev_centroids(i, :);
    else
        points = X((memberships == i), :);
        centroids(i, :) = mean(points);    
    end
end

end
