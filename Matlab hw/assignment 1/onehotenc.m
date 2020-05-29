function [d] = onehotenc(nclasses, k)
%ONEHOTENC return a onehot vector of class k
%   vector with zeros and 1 at position k
    I = eye(nclasses);
    d = I(:, k);
end