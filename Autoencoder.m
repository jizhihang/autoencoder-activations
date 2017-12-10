%% CS 532 Final Project -- Rita Roloff, Justin Essert, Aaron Levin

%% Load data
train_images = loadMNISTImages('MNIST/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('MNIST/train-labels-idx1-ubyte');

test_images = loadMNISTImages('MNIST/t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('MNIST/t10k-labels-idx1-ubyte');

% Training and testing images are known to be 28x28 from the dataset
% specifications, but, the manner in which we are loading them puts them
% into a neural-network-friendly 2D format of: 
%   784x60000 (training images)
%   784x10000 (testing images)
%   60000x1   (training labels)
%   10000x1   (testing labels)

%% Setup hyperparameters

max_epoch = 10000; % number of training iterations to run
epsilon = .001; % convergence factor
alpha = 0.01; % step size
num_hidden = 784;
[m, n] = size(train_images); % input layer dimensions
t = 0;
err = Inf; % initialize to some value that clearly has not converged yet
act_func = 1; % which activation function to use

%% Train Autoencoder
while( (err > epsilon) && (t < max_epoch) )
    
    % Pick random image non-mutually-exclusively
    img_idx = round(rand(1)*n);
    x = train_images(:,img_idx)';
    
    % Generate normally distributed weight matrices with pseudorandom numbers.
    w = randn(m, num_hidden); % weights for inputs-hidden
    v = randn(num_hidden, m); % weights for hidden-outputs

    % Forward Propogation
    h_p = x * w;
    h = act(h_p, act_func);
    o_p = h * v;
    o = act(o_p, act_func);

    % Backpropagation
    deltaoh = (o - x)*deriv(o_p, act_func);
    vbp = v - alpha * deltaoh * h';
    deltahi = deltaoh * v * deriv(h_p, act_func);
    wbp = w - alpha * deltahi * x';
    
    % Update Error
    err = norm(vbp - v);
    
    % Update weights
    w = wbp;
    v = vbp;
    
    % Go to next iteration
    t = t + 1; 
end

%% Test Autoencoder

% Training Set Error
yhat = act(act(train_images' * w, act_func) * v, act_func);
error = norm(yhat - train_images')


