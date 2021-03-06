function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:length(X);
  dist = inf;
  test_x = X(i,:);
 for j = 1:K
    test_centroid = centroids(j,:);
    size(test_centroid);
    size(test_x);
    test_dist = sum((test_x-test_centroid).^2);
    if test_dist <dist,
      dist = test_dist;
      idx(i) = j;
    end
   end
end
%dist_mat = zeros(size(X,1),K)
%  for i =1:K
%    dist_mat(:,i) = (X - centroids(i,:)).^2
%    end
    
  





% =============================================================

end

