function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = length(grad);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Could also grab costFunction definitions and just add. Dunno?
hypothesis = sigmoid(X*theta);
J = (1/m) * sum(-y.*log(hypothesis) - (1-y).*(log(1-hypothesis))) + (lambda/2/m)*sum([0; theta(2:end)].^2);

% Turns out, looping to do this vector.*Xcolumn operation is about as fast 
%   as it gets, due to MATLAB's JIT compilation. So don't bother
%   vectorizing
for jj = 1:n
    grad(jj) = (1/m) * sum((hypothesis - y).*X(:,jj))';
end
grad = grad + (lambda/m)*[0; theta(2:end)];
% =============================================================

end
