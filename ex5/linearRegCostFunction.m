function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

hypothesis = X * theta;

% size(hypothesis)
% size(Y)
% keyboard();

xydiff = hypothesis .- y;
err = sum(xydiff .^ 2);

thetasq = theta .^ 2;
% don't regularize theta_0
thetasq(1) = 0;
reg = lambda * sum(thetasq);

J = err + reg;
J = J / (2 * m);

% calculate gradients
for j=1:size(theta)(1),
    % keyboard();
    grad(j) = sum(xydiff .* X(:, j));
end

grad = grad / m;

% gradient regularization term
% don't regularize theta_0
for j = 2:size(theta)(1),
    grad(j) = grad(j) + (lambda / m) * theta(j);
end


grad = grad(:);

end
