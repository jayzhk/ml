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
Theta2_grad = zeros(size(Theta2));

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

%   --------------------  Part 1-1  compute J without regularization -----------------------

% Convert y in to the matrix Y 

Y = zeros(m, num_labels);

for i= 1:m,
     Y(i, :) = (y(i) == [1:num_labels]);
end

X = [ones(m, 1) X];
A2=[ones(m, 1) sigmoid(X * Theta1')];
A3 = sigmoid(A2 * Theta2');
J= -sum(sum( Y .* log(A3) + (ones(m, num_labels) - Y) .* log(1 - A3))) /m ;

% --------------------Part 1 - 2 compute J with regularization -----------

Theta1_unbias = Theta1(:, 2:size(Theta1, 2));
Theta2_unbias = Theta2(:, 2:size(Theta2, 2));

reg = (lambda / (2 * m) ) * (sum(sum(Theta1_unbias .^2)) + sum(sum(Theta2_unbias .^2)));
J = J + reg;

% ======================== Part 2  backpropagation ========================
D1 = zeros(hidden_layer_size, input_layer_size + 1);
D2 = zeros(num_labels , hidden_layer_size +1);
D3 = zeros(num_labels,1);
for i = 1:m,
    % step 1
    a_1 = X(i, :)';
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    % step 2
    delta_3 = a_3 - (y(i) == [1:num_labels])';
    delta_2= (Theta2' * delta_3) .*[1; sigmoidGradient(z_2)];
    delta_2 = delta_2(2:size(delta_2));
    
    % step 3
    D1 = D1 + delta_2 * a_1';
    D2 = D2 + delta_3 * a_2';
    D3 = D3 + delta_3;
end

% ===== below is the un-regularized partial derivative ====================
% Theta1_grad = D1 /m;
% Theta2_grad = D2 /m;

% =================== Part 3 - regularized back propagation ===============

Theta1_grad = [D1(:, 1)  (D1(:, 2:size(D1, 2)) + lambda * Theta1(:, 2:size(Theta1, 2)))]/m; 
Theta2_grad = [D2(:, 1)  (D2(:, 2:size(D2, 2)) + lambda * Theta2(:, 2:size(Theta2, 2)))]/m; 

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
