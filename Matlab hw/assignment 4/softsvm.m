function [a, b] = softsvm(X, t, C)
  
N = size(X,1);
K = X*X'.*(t*t');
H = K;
f = ones(N,1); % 1T
A = [];
b = [];
LB = repmat(0,N,1);
UB = repmat(C,N,1); % refer to documentation


alpha = quadprog(H,-f,A,b,t',0,LB,UB); % Following line runs the SVM
    
    % Compute bias
fout    =  X*X'*alpha.*t; % to be used in bias calculation

pos     = find(alpha<C);
bias    = mean(t(pos)-fout(pos));

a = alpha;
b = bias;
end