function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% DOES NOT ACCOUNT FOR SINGLE FEATURES FOR NOW

% You need to set these values correctly
X_norm = X;
mu = mean(X);
sigma = std(X);
% in case no std...
sigma(sigma == 0) = 1;

for feature = 1:size(X,2)
    X_norm(:,feature) = (X(:,feature)-mu(feature))/sigma(feature);
end


% Reset
sigma(sigma==1)=0;
% ============================================================

end
