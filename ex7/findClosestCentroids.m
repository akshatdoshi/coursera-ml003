function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);


% You need to return the following variables correctly.
idx = zeros(size(X, 1), 1);

m = size(X, 1)
n = size(centroids, 1)

for i=1:m % for each example
    % dist is a column vector containing the distance of this point (X(i)) to each centroid
    dist = zeros(n, 1);

    point = X(i, :)'; % column vector

    for k=1:n
        cent = centroids(k, :)'; % centroid that we're comparing against as a column vector

        for dim=1:size(point)(1)
            dist(k) = dist(k) + (point(dim) - cent(dim)) ^ 2;
        end
    end


    [x, index] = min(dist);
    idx(i) = index;
end

end

