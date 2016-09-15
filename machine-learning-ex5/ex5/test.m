%% Initialization
clear ; close all; clc

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

theta = [1 ; 1];

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
lambda = 1;
X = [ones(m, 1) X];
Xval = [ones(size(Xval, 1), 1) Xval];
% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

%J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

%fprintf(['Cost at theta = [1 ; 1]: %f '...
%         '\n(this value should be about 303.993192)\n'], J);