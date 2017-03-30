function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

J = 0.0;
hypothesis = sigmoid(X * theta);
first = -y .* log(hypothesis);
second = (1.0 .- y) .* log(1.0 .- hypothesis);

unregularized = (1.0 / m) * sum(first - second);
regularization = (lambda / (2.0*m)) * sum(theta(2:length(theta),:) .^ 2);
J = unregularized + regularization;

grad = zeros(size(theta));
grad = grad(:);

% Don't regularize the first theta
beta = theta;
beta(1,1) = 0;

hypothesis = sigmoid(X * theta);
unregGrad = (1/m)*sum((hypothesis - y) .* X);
reg = beta * (lambda/m);
grad = transpose(unregGrad) + reg

end
