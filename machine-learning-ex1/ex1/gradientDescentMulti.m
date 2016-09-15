function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % ============================================================
     %calculate hyp with new theta values
    h = X*theta;
    
    %calculate error and copy result into new row
    %cost_partial = [h-y, h-y]';
    cost_partial = repmat((h-y)', size(theta, 1), 1);

    %row0 is theta-0, row1 is theta-1
    sum_partial = sum(cost_partial.*X',2)/m;
    
    %take a step and update simultaneously
    theta = theta - (alpha * sum_partial);
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
