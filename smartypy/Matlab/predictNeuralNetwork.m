function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add the offset column
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%% Layer 2 Computations
z2 = X * Theta1';
a2 = [ones(size(z2,1),1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

h = a3; %possibilities

% possibilities now contains hypothesis value for each classicification in
% a separate column.
%
% [  (prob row1==1), (prob row1==2), (prob row1==3)...]
% The index and result are coincident here. So if the maximum probablity
% lies in column 7, the suggeested classification is in fact 7.

% Find value and index (p) of maximum of each probablity per row
[prob,p] = max(h,[],2);

% =========================================================================


%==========================================================================
% smaller, less explicit version
% m = size(X, 1);
% num_labels = size(Theta2, 1);

% p = zeros(size(X, 1), 1);

% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');
% [dummy, p] = max(h2, [], 2);

end
