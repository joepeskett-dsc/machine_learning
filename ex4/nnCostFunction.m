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

Delta_1 = 0;
Delta_2 = 0  ;
  
for t = 1:m;
    %1 Perform a feedforward pass
    a_1 = [1 X(t,:)];
    z_2=(a_1 * Theta1');
    a =(sigmoid(z_2));
    m =size(X,1);
    a_2 = [1 a];
    a_3 = sigmoid(a_2 * Theta2');
    h =a_3;
    
    %2 set output of d_3 to difference between prediction and Y value
    Y_vec = 1 : num_labels;
    Y = y(t,:) == Y_vec;
    d_3 = h - Y;
    
    %3 set d_2
    size(Theta2);
    size(d_3)
    size(z_2);
    d_2= (d_3*Theta2).*  [1 sigmoidGradient(z_2)];
    size(d_2)
    d_2 = d_2(2:size(d_2);
    size(d_2)
    
    %4accumulate the gradients
    
    Delta_2 = Delta_2 + (d_3'*(a_2));
    Delta_1 = Delta_1 + (d_2'*(a_1));
   
  end
  %5
    Theta1_grad = Delta_1/m;
    Theta2_grad = Delta_2/m;
 
%grad=((1/m)*((h-y)'*X)');









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
