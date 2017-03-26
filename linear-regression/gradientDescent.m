function [theta, J_history] = gradientDescent(X, y, theta, a, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	tmpTheta = zeros(2,1);

	tmpTheta(1) = theta(1) - ((a/m) * sum(((theta(1) + theta(2) * X(:, 2)) .- y) .* X(:, 1)));
	tmpTheta(2) = theta(2) - ((a/m) * sum(((theta(1) + theta(2) * X(:, 2)) .- y) .* X(:, 2)));

	theta = tmpTheta;

    J_history(iter) = computeCost(X, y, theta);
end

end
