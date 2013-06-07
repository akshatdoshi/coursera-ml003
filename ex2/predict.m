function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

for i=1:m,
    %h = sigmoid(theta' * X(i,:)'); % the h_theta(x^(i)) term for this iteration
    %h = sigmoid(X(i,:)'' * theta); % the h_theta(x^(i)) term for this iteration

    row = X(i,:);
    %row = [1 45 85]; % TODO remove

    h = sigmoid(row * theta);

    % printf('X rowa = \n');
    % disp(row);

    % printf('theta = \n');
    % disp(theta);

    %printf('h = \n');
    %disp(h);

    if (h >= 0.5),
        prediction = 1;
    else
        prediction = 0;
    endif

    %printf('h = %f, p = %f\n', h, prediction);

    p(i, 1) = prediction;
end

%printf('p = \n');
%disp(p);

end
