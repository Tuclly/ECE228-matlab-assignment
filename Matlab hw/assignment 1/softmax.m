function [y] = softmax(z)
% z is a K x 1 vector of floats.
% y should be the output of softmax function
m = max(z);
[y] = exp(z - m)/sum(exp(z - m));%为了符合sf(z) = sf(z + c)
end


