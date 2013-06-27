function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

Cvec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmavec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

bestsigma = sigmavec(1);
bestcost = 1000000; % blech
bestc = Cvec(1);

for i=1:length(Cvec)
    c = Cvec(i);
    for j=1:length(sigmavec)
        sigma = sigmavec(j);
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        cost = mean(double(predictions ~= yval));

        if cost < bestcost
            bestcost = cost;
            bestsigma = sigma;
            bestc = c;
        end

        printf('tested c=%f, sigma=%f, cost=%f, bestc=%f, bestsigma=%f, bestcost=%f\n', c, sigma, cost, bestc, bestsigma, bestcost);
    end
end

C = bestc;
sigma = bestsigma;

end
