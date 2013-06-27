function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

sumterm = 0;

for k=1:length(x1)
    sumterm = sumterm + (x1(k) - x2(k)) ^ 2;
end

sim = exp(-sumterm / (2 * sigma ^ 2));

end
