function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
delta3 = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%% Forward Propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Recode the y vector of labels to a vector representation i.e. label 1 is
% [0, 1, 0, ..., 0]
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% Define a1, the input layer (X) with the bias units
a1 = [ones(m,1), X];

% Calculate the sigmoid argument
z2 = a1*Theta1';

% Calculate the single hidden layer hypotheses (25 neurons + 1 bias unit)
a2 = sigmoid(z2);
a2 = [ones(m,1), a2];

% Calcuate the sigmoid argument
z3 = a2*Theta2';

% Calculate the output layer hypotheses
a3 = sigmoid(z3);

% Calculate the left and right side terms of the cost function
% Ten columns represent each output unit (K=10)
l = -y_matrix.*log(a3);
r = (ones(size(y_matrix)) - y_matrix).*log(ones(size(a3)) - a3);

% Sum up each hypotheses from each output units (num_labels, 0-9) [5000x10]
% Sum each column up [1x10]
sum_output_hypotheses = sum(l-r);

% Unregularized cost function
% Sum up and average the hypotheses
J = sum(sum_output_hypotheses)/m;

% Sum squared-paramaters for layer a1 to a2 for each hidden node (25x401)
% Sum up theta1-squared value to average later
Theta1_features_sum = sum(Theta1(:, 2:end).^2);
Theta1_final_sum = sum(Theta1_features_sum);

Theta2_features_sum = sum(Theta2(:, 2:end).^2);
Theta2_final_sum = sum(Theta2_features_sum);

% 1/2m to average both sums
regulator = (lambda/(2*m))*(Theta1_final_sum + Theta2_final_sum);

% Average the cost values
J = J + regulator;

%%%%%%%%%%%%%%%%%%%%%%% Backward Propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the error term for the output layer (a3)
sigma3 = a3-y_matrix;

% Add bias units to z2 and calculate delta 2
% Discard the delta2 bias unit
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(length(z2),1), z2]);
sigma2 = sigma2(:,2:end);

% Calculate the parameter gradients
delta2 = (sigma2'*a1)/m;
delta3 = (sigma3'*a2)/m;

% Regularized gradients, ignore bias units from regularization
% Replace bias units with zeros
Theta1_reg = (lambda/m)*[zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
Theta2_reg = (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

Theta1_grad = delta2 + Theta1_reg;
Theta2_grad = delta3 + Theta2_reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
