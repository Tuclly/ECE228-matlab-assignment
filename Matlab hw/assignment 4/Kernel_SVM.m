function [a, b] = softsvm_RBF(X, t, C)

N = size(X,1);
nms = sum(X'.^2,1);
K = exp(-(nms'*ones(1,N) + ones(N,1)*nms - 2*X*X')).*(t*t');
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