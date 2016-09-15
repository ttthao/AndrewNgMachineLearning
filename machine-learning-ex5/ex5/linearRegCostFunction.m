function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X*theta;
cost = sum((h - y).^2);
penalty = lambda*sum((theta(2:length(theta)).^2));
J = (cost + penalty)/(2*m);

% Duplicate cost to avoid regularizing theta0
% Row 0 is theta0, row 1 is theta 1
% [ h-y ; h-y ]
cost_partial = repmat((h-y)', size(theta, 1), 1);

% Element-wise multiplication of X and row sum
grad = sum(cost_partial.*X',2)/m;

% Regularize theta1 and on
grad(2:length(theta)) = grad(2:length(theta)) + (lambda/m).*theta(2:length(theta));
% =========================================================================

grad = grad(:);

end
