function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
for n = 1:m
	x = transpose(X(n,:));
	currentY = y(n);
	hypothesis = sigmoid(transpose(theta) * x);

	first = -currentY * log(hypothesis);
	second = (1 - currentY) * log(1 - hypothesis);
	J += (1/m) * sum(first - second);
end

grad = zeros(1,3);

for i = 1:length(theta)
	for n = 1:m
		x = transpose(X(n,:));
		currentY = y(n);
		hypothesis = sigmoid(transpose(theta) * x);

		grad(1, i) += (1/m) * (hypothesis - currentY) * X(n, i)
	end
end

end
