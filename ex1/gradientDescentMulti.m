function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X)(1,2);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    hyp = X * theta;
    %printf('hyp = %f\n', hyp)
    for i = 1:n,
        err = hyp - y;
        grad(i, 1) = sum(err .* X(:,i)) / m;
    end

    %printf('grad=%f\n', [grad]);
    newtheta = theta - (alpha .* grad);
    theta = newtheta;
    %printf('theta=%f\n', theta)

    J = computeCostMulti(X, y, theta);
    %printf('iter=%d J=%f grad=%f, nt=%f\n', iter, J, grad, newtheta);
    %printf('iter=%d J=%f\n', iter, J);

    % Save the cost J in every iteration    
    J_history(iter) = J;

end

end
