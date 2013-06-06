function g = sigmoid(z)
    %SIGMOID Compute sigmoid functoon
    %   J = SIGMOID(z) computes the sigmoid of z.

    % You need to return the following variables correctly 
    g = zeros(size(z));

    for row = 1:size(z)(1),
        for col = 1:size(z)(2),
            g(row,col) = 1 / (1 + exp(-z(row,col)) )
        end
    end
end
