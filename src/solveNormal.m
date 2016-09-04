function [theta] = solveNormal(X, y)
%SORVENORMAL Computes the closed-form solution to linear regression 
%   SOLVENORMAL(X,y) computes the closed-form solution to linear 
%   regression using t\ normal equations.

% theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

theta = pinv(X'*X)*X'*y;


% ============================================================

end
