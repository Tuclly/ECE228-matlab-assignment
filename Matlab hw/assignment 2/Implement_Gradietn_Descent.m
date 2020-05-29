%% Assessment 2
clear
rng('default');

%% loading data
load mnist.mat

% rename label 0 to 10
train_labels(train_labels == 0) = 10;
test_labels(test_labels == 0)   = 10;
labels = unique(train_labels);

%% Neural Network

% Fixed parameters
d = size(train_data, 2); % MNIST digit size 
nclasses = length(labels); % total number of classes
Ni = d; % Number of external inputs
Nh = 200; % Number of hidden units
No = nclasses; % Number of output units
alpha_i = 0.0; % Input weight decay
alpha_o = 0.0; % Output weight decay
range = 0.1; % Initial weight range                
eta=0.001; % gradient descent parameter

% Initialize network weights
Wi = range * randn(Nh,Ni+1);
Wo = range * randn(No,Nh+1);

max_iter=5;             % maximum number of iterations
iter = 1;
fprintf('Training ...\n');



while iter < max_iter
  fprintf('Iteration %d ...\n', iter);
  % implement gradient descent updates here
  % hint use fullGradient
  [dWi, dWo] = fullGradient(Wi,Wo,alpha_i,alpha_o, train_data, train_labels, nclasses);
  Wi = Wi - eta*dWi;
  Wo = Wo - eta*dWo;
  
  
  
  
  iter = iter + 1;
end



% Test and print accuracy
fprintf('Testing ...\n');
acc = 0;
N   = length(test_labels);N=5;
y_pred = zeros(size(test_labels));
for k = 1:N
  yi = [1;train_data(k, :)']; % input
  v1 = Wi*yi; % FC
  yh = [1;relu(v1)]; % hidden layer w/ bias
  
  % output layer
  v2 = Wo*yh; % FC
  yo = softmax(v2); % softmax
  
  [~, i] = max(yo);
  y_pred(k) = i;
  if i == train_labels(k)
    acc = acc + 1;
  end
  if(rem(k, 100)==0)
    fprintf('%d done.\n', k);
  end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);

function [y] = softmax(z)
% paste your softmax function here
m = max(z);
[y] = exp(z - m)/sum(exp(z - m));%为了符合sf(z) = sf(z + c)
end
    
function [y] = relu(x)
% paste your relu function here
[y] = max(0,x);
end

function [d] = onehotenc(nclasses, k)
% paste your one hot encoder here
    I = eye(nclasses);
    d = I(:, k);
end

function [dWi,dWo] = fullGradient(Wi,Wo,alpha_i,alpha_o, Inputs, Labels, nclasses)
% Paste your code for calculating gradients here
%
% Calculate the partial derivatives of the quadratic cost function
% wrt. the weights. Derivatives of quadratic weight decay are included.
%
% Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
% Output:
%        dWi     :  Matrix with gradient for input weights
%        dWo     :  Matrix with gradient for output weights

% Determine the number of examples

    function [y] = softmax(z)
    % Paste your softmax function here
        m = max(z);
        [y] = exp(z - m)/sum(exp(z - m));%in order to satisfy sf(z) = sf(z + c)
    end
    
    function [y] = relu(x)
    % Paste your relu function here
        [y] = max(0,x);
    end
    
    function [d] = onehotenc(nclasses, k)
    % Paste your one hot encoder here
    I = eye(nclasses);
    d = I(:, k);
    end

[ndata, inp_dim] = size(Inputs);

dWi = zeros(size(Wi));
dWo = zeros(size(Wo));

% compute the derivatives for each data point separately.
for k=1:ndata
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  %
  % Propagate kth example forward through network
  % calculating all hidden- and output unit outputs
  %
  
  % Calculate hidden unit outputs for every example
  x = [1,Inputs(k,:)]'; % 785 * 1 
  g = Wi * x; % 200*1
  
  h = [1;relu(g)];
  y = softmax(Wo*h); % 10 *1

  
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%
  % Back propagation
  % Calculate derivative of
  % by backpropagating the errors from the
  % desired outputs
  t = onehotenc(nclasses, Labels(k));
  Do = y - t;
  Di = (g>0) .* (Wo(:,2:201)'* Do);
  dWo = dWo + Do * h';
  dWi = dWi + Di * x';

end


% Add derivatives of the weight decay term
dWo = dWo/ndata + alpha_o*Wo;
dWi = dWi/ndata + alpha_i*Wi;
end