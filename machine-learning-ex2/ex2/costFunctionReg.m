function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = sigmoid(X * theta);

first = -y .* log(hypothesis);
second = (1 .- y) .* log(1 .- hypothesis);

unregularized = (1/m) * sum(first .- second);
regularization = (lambda / (2*m)) * sum(theta(2:length(theta),:));
J = unregularized + regularization;

% =============================================================

% Don't regularize the first theta
regTheta = theta;
regTheta(1,1) = 0;

for i = 1:length(theta)
	hypothesis = sigmoid(X * theta);
	unregGrad = (1/m)*sum((hypothesis - y) .* X(:, i));
	reg = regTheta(i) * (lambda/m);
	grad(i, 1) = unregGrad + reg;
end

end
