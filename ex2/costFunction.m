function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
nFeatures = size(X)(1,2); % number of features, including 1 const col

% printf('X:');
% disp(X);

% theta is a (nFeatures, 1) matrix

J = 0;
grad = zeros(size(theta));
% printf('grad %d:\n', nFeatures);
% disp(size(grad));

% print('hyp\n');
% disp(hyp);

for i = 1:m,
    %h = sigmoid(theta' * X(i,:)'); % the h_theta(x^(i)) term for this iteration
    h = sigmoid(X(i,:) * theta); % the h_theta(x^(i)) term for this iteration
    %printf('h = \n');
    % disp(h);

    J = J + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
    for j = 1:nFeatures;
        grad(j, 1) = grad(j, 1) + (h - y(i)) * X(i, j);
    end

end

J = J / m;
grad = grad / m;

% printf('J\n');
% disp(J);

% printf('grad\n');
% disp(grad);

end
