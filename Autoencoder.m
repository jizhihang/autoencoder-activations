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

max_epoch = 100; % number of training iterations to run
epsilon = .0001; % convergence factor
alpha = 0.01; % step size
num_hidden = 784;
[m, n] = size(train_images); % input layer dimensions
t = 0;
err = Inf; % initialize to some value that clearly has not converged yet
act_func = 2; % which activation function to use

Xb = [ones(size(train_images,1)) train_images];

% Generate normally distributed weight matrices with pseudorandom numbers.
w = randn(m+1, num_hidden); % weights for inputs-hidden
v = randn(num_hidden+1, m); % weights for hidden-outputs


%% Train Autoencoder
while( (err > epsilon) && (t < max_epoch) )
    
    % Pick random images
    img_idx = randperm(n);

    for i = img_idx ;
               
        x = Xb(:,i)';
    
        % Forward Propogation
        h_p = [1 x] * w;
        h = act([1 h_p], act_func);
        o_p = h * v;
        o = act(o_p, act_func);

        % Backpropagation
        deltaoh = (o - x).*deriv(o, act_func);
        vbp = v - alpha * h' * deltaoh;
        deltahi = deltaoh * v(2:end, :)' .* deriv(h(2:end), act_func);
        wbp = w - alpha * deltahi * x';

        % Update Error
        err = norm(vbp - v)

        % Update weights
        w = wbp;
        v = vbp;
        
    end
    
        % Go to next epoch
        t = t + 1; 
end

%% Test Autoencoder

% Training Set Error
h_p = [1 x] * w;
h = act([1 h_p], act_func);
o_p = h * v;
o = act(o_p, act_func);

err = norm(o - x)



%yhat = act(act(train_images' * w, act_func) * v, act_func);
%error = norm(yhat - train_images')


