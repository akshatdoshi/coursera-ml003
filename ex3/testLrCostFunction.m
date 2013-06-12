clear ; close all; clc

data = load('../ex2/ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% options = optimset('GradObj', 'on', 'MaxIter', 10);
options = optimset('GradObj', 'on', 'MaxIter', 800);

% Compute and display initial cost and gradient
initial_theta = zeros(n + 1, 1);
[cost, grad] = ex2costFunction(initial_theta, X, y, 1);
printf('ex2CostFunction cost = \n');
disp(cost);
printf('ex2CostFunction grad = \n');
disp(grad);
[theta, cost] = fminunc(@(t)(ex2costFunction(t, X, y, 1)), initial_theta, options);
printf('ex2CostFunction theta\n');
disp(theta);
printf('ex2CostFunction after cost\n');
disp(cost);

printf('ex2CostFunction after grad\n');
J, grad = ex2costFunction(theta, X, y, 1);
disp(J);
disp(grad);


initial_theta = zeros(n + 1, 1);
[cost, grad] = lrCostFunction(initial_theta, X, y, 1);
printf('\nlrCostFunction cost = \n');
disp(cost);
printf('lrCostFunction grad = \n');
disp(grad);
[theta, cost] = fminunc(@(t)(lrCostFunction(t, X, y, 1)), initial_theta, options);
printf('lrCostFunction theta\n');
disp(theta);
printf('lrCostFunction after cost\n');
disp(cost);

printf('lrCostFunction after grad\n');
J, grad = lrCostFunction(theta, X, y, 1);
disp(J);
disp(grad);

