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

% Part 1 - Forward propogation
y_mat = zeros(m, num_labels);

for i = 1:m;
  y_vec =(1:num_labels);
  y_mat(i,:) = (y_vec == y(i));
  
end

size(y_mat);
a_1 = [ones(m, 1) X];
a =(sigmoid(a_1*Theta1'));
m =size(X,1);
a_2 = [ones(m,1) a];
a_3 = sigmoid(a_2 * Theta2');
h =a_3;
y =y_mat;
J_unreg = sum(sum((-y.* log(h) - (1 - y) .* log(1-h))))'*(1/m);
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;

Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
regular = (lambda/(2*m))*(sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));
J = J_unreg + regular;






%This is cost function for logistic regression
%h=sigmoid(X*theta);
%shit_theta = (theta(2:size(theta)));
%reg_theta = [0; shit_theta];
%J=sum((-y.* log(h) - (1 - y) .* log(1-h)))*(1/m)+ ((lambda / (2*m))*sum(reg_theta.^2));
%grad=((1/m)*((h-y)'*X)') + ((lambda/m) * reg_theta);


% Part 2 Back Propogation

%1 set input layer value to a_1 and perform feedforward pass
Delta_1 = 0;
Delta_2 = 0  ;
for t = 1:m

  a_1 = [1;X(t,:)'];
  z_2 = Theta1*a_1;
  a_2 = sigmoid(z_2);
  a_2 = [1; a_2];
  z_3 = Theta2*a_2;
  a_3 = sigmoid(z_3);
  h = a_3;
  size(h);
 %2 for each output unit k, set d=a_3-Y (Y should be a vector) 

  Y = y_mat(t,:);
  d_3 = h - Y';
    
%3 For hidden layer (l=2) set d_2 = Theta2'*d_3 .*g'(z_2)

  d_2 = Theta2'*d_3.*[1; sigmoidGradient(z_2)];
  size(d_2);
  
%4 Accumulate the gradients - remember to remove d0_2.

  d_2 = d_2(2:end);

  Delta_1 += d_2*a_1' ;
  Delta_2 +=  d_3*a_2' ;
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
 
%Delta_1(:,2:end) = Delta_1(:,2:end) + lambda * (1/m) * Theta1(:,2:end);
%Delta_2(:,2:end) = Delta_2(:,2:end) + lambda * (1/m) * Theta2(:,2:end);


Theta1_grad = Delta_1*(1/m)+ (lambda * (1/m) * Theta1(:,2:end));
Theta2_grad = Delta_2*(1/m)+ (lambda * (1/m) * Theta2(:,2:end));


grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
