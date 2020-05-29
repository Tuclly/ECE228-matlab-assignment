function [x] = OMPnorm(A,y,K)
% Orthogonal Matching Pursuit (OMP)
% Solve y = Ax, assuming x is a sparse vector with sparsity numOMP.
% [x] = OMPnorm(A,y,numOMP)
% [INPUTS]
% A - dictionary (N x M)
% y - Data vector of size N x 1
% K - desired sparsity (number of atoms)
% [OUTPUTS]
% x - Sparse coefficient vector of size M x 1

[N, M] = size(A);


Anorm = A; %normalize dictionary A such that all columns have norm equal to 1

x = zeros(M,1);
indx = zeros(K,1);
A_cols = [];

residual = y;                           % initial residual is full measurement vector
for j = 1:K
    x_tmp = zeros(M,1);
    proj     = Anorm'*residual;%calculate dot product between residual and each column of Anorm 
   
    inds = setdiff([1:M],indx);
    for i = inds
        x_tmp(i) = A(:,i)' * residual / norm(A(:,i)); 
    end
%     [~,pos] = max(abs(x_tmp));
    
    [~,pos]  = max(proj);%find index number of maximum value of proj
    pos      = pos(1);                  % choose first value if multiple correlations are equal
    indx(j)  = pos;                     % store indices
    
    A_cols = [A_cols A(:,pos)];                                
    a        = A_cols \ y;%least squares estimate of x using only columns of A given by indx
    residual = y - A_cols * a;%residual = signal minus projection
end

x(indx(1:j)) = a;