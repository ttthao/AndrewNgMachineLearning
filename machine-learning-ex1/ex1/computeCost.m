function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%
% NOTES
%
% 1. X is the whole set of x-values of the giventraining set. (97x2)
% 2. theta is the vector of the 2 theta values. (2x1)
% 3. X*theta is vector of the result of the current theta values & the x-value
% matrix (97x1)
% 4. X* theta - y is subtracting the result vector by the y-value vector to
% compute the cost (97x1)
% 5. .^2 does element-wise squaring & the sum() sums every row of the cost
% vector (97x1)
% 6. J is the final result value (1x1)

J = sum((X*theta - y).^2)/(2*m);
% =========================================================================

end
