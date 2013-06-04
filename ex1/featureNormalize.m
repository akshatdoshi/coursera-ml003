function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

for i=1:size(X)(1,2),
    printf('feature %d\n', i);
    thismu = mean(X(:,i));
    thissigma = std(X(:,i));
    printf('mean = %f, sd = %f\n', thismu, thissigma);
    X_centred(:,i) = X(:,i) - thismu;
    X_norm(:,i) = X_centred(:,i) / thissigma;

    mu(1, i) = thismu;
    sigma(1, i) = thissigma;

    %fprintf(' after x = [%.1f %.1f]\n', [X(1:10,:)]');
end


% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by its standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
