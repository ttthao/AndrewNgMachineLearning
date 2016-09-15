function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    
    %calculate hyp with new theta values
    h = X*theta;
    
    %calculate error and copy result into new row
    %cost_partial = [h-y, h-y]';
    %cost_partial = repmat((h-y)', 2, 1);
    %Better Version
    cost_partial = repmat((h-y)', size(theta, 1), 1);

    
    %row0 is theta-0, row1 is theta-1
    sum_partial = sum(cost_partial.*X',2)/m;
    
    %take a step and update simultaneously
    theta = theta - (alpha * sum_partial);   
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    % debug
    %fprintf('J_history(iter) is %f at iter %d \n', J_history(iter), iter);
    %fprintf('theta is %f %f \n', theta(1), theta(2));
end

end
