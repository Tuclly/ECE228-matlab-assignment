function [y] = relu(x)
%RELU implements the relu activation function.
    [y] = max(0,x);

end

% x = linspace(-5, 5);
% 
% y = relu(x);
% 
% plot(x, y);
% title("ReLU");