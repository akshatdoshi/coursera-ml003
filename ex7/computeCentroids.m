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

if 0
    disp(K)
    printf('...\n');
    disp(idx)
    printf('===\n');
    disp(centroids)
    printf('---\n');
    disp(X)
    printf(',,,\n');
end


% for each centroid
%   create a flag vector 1/0 where it's set if we're doing that centroid (elem in idx == loop iter)
%   mutliply flag vector by ecah feature of X
%   
%   divide each cell in total by number of flags set

flags = zeros(m, K);
for i=1:m
    flags(i, idx(i,1)) = 1;
end

for i=1:K
    flagvec = flags(:, i)';

    % number of samples assigned to this centroid
    nsamples = sum(flagvec);

    % compute mean
    centroids(i, :) = (flagvec * X) ./ nsamples;
end

end

