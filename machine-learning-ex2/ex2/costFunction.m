function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
%fprintf(transpose(theta) .* X)

hypothesis = sigmoid(transpose(theta) .* X);
oneCase = -y .* log(hypothesis);
zeroCase = (1 .- y) .* log(1 .- hypothesis);
J = (1/m) .* sum(zeroCase .- oneCase);

hypothesisTakeY = hypothesis .- y;
multiplyThetaRow = hypothesisTakeY .* X(:, 2);
averagedSum = 1/m .* sum(multiplyThetaRow);
grad(1,1) = averagedSum(1,2);


multiplyThetaRow = hypothesisTakeY .* X(:, 3);
averagedSum = 1/m .* sum(multiplyThetaRow);
grad(1,2) = averagedSum(1,3);

end
