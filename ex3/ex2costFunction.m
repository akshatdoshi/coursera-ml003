function [J, grad] = ex2costFunction(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
nFeatures = size(X)(1,2); % number of features, including 1 const col

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

totalreg = 0.0;

for i = 1:m,
    h = sigmoid(X(i,:) * theta); % the h_theta(x^(i)) term for this iteration

    regsum = 0.0; % the sum term of the regularization term
    for j = 2:nFeatures, % start at 2 so as to skip parameter theta_0
        regsum = regsum + theta(j, 1) ^ 2;
    end
    regterm = regsum * lambda / (2 * m);
    %regterm = regsum * lambda / (2);
    totalreg += regterm;


    % regterm = 0.0; % NOTE disable regularization

    J = J + (-y(i) * log(h) - (1 - y(i)) * log(1 - h)) + regterm;
    for j = 1:nFeatures,
        grad(j, 1) = grad(j, 1) + (h - y(i)) * X(i, j);
    end
end

% printf('regterm = ');
% disp(regterm);

J = J / m;
grad = grad / m;

% add the regularization terms to grad
for j = 2:nFeatures, % again, start at 2 so as to not regularize theta_0
    grad(j, 1) = grad(j, 1) + theta(j, 1) * lambda / m;
    gradreg(j,1) = theta(j, 1) * lambda / m;
end
% printf('ex2 grad reg term\n');
% disp(gradreg');

end
