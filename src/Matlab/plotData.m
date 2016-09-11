function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
plot(X(y==0,1),X(y==0,2),'ko','MarkerSize',7,'MarkerFaceColor','y');
plot(X(y==1,1),X(y==1,2),'k+','MarkerSize',7,'LineWidth',2);
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Not Admitted','Admitted')









% =========================================================================



hold off;

end
