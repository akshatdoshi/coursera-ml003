function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

H = sigmoid(X * theta); % the h_theta(x^(i)) term for this iteration

% cost
term1 = -y .* log(H);
oneminusH = 1 - H;
logH = log(oneminusH);
oneminusY = 1 - y;
term2 = oneminusY .* logH;
termsum = term1 - term2;
J = sum(termsum) / m;

% regularization term
thetasquared = theta .^ 2;
thetasquared(1) = 0;
reg = sum(thetasquared) * (lambda / (2 * m));
J = J + reg;

% calculate the gradients
betai = H - y;
grad = X' * betai / m; % the summation at the bottom of page 6 of ex3.pdf

% regularization term
temp = theta;
temp(1) = 0;
gradreg = (lambda / m) .* temp;
grad = grad + gradreg;

end
