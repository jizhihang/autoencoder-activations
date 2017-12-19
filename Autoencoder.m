%% CS 532 Final Project -- Rita Roloff, Justin Essert, Aaron Levin
close all; clear all;
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

% Take 10% of training set and use it for validation
x = train_images';
indices = crossvalind('Kfold', ones(1, size(x, 1)), 10);
x_train = x(~(indices == 1),:);
x_val = x(indices == 1,:);


%% Setup hyperparameters

num_hidden = 100;   % number of hidden 
act_func = 1;       % activation function
% alpha = 1e-3;       % step size
% lambda = 0;      % regularization parameter
batch = 1000;        % batch size
max_epoch = 25;      % number of training iterations to run

%% Train Network
lambda = 10.^(2:-2:-4);
alpha = 10.^[1:-1:-4  -6:-2:-10];
losses = zeros(size(lambda,2), size(alpha,2)); % rows are lambda, columns are alpha

for j=1:5
    act_func = j
    min = Inf;
    min_idx = [0  0];

    % figure(1); hold on;
    for i=1:size(lambda,2)
        for k=1:size(alpha, 2)
            disp(['lambda = ',num2str(lambda(i)),', alpha = ', num2str(alpha(k))])
            [ w, v, loss ] = train_network(x_train, x_train, x_val, x_val, num_hidden, act_func, alpha(k),lambda(i), batch, max_epoch );
            t = size(loss,2);
            if(t~=max_epoch)
                % Had to break, values became nonreal
                loss(t+1:max_epoch) = loss(t);
            end

            if loss(t) < min, min = loss(t); min_idx = [i k]; end

            % plot(1:max_epoch,loss);
            % leg(i) = "lambda = "+num2str(lambda(i));
        end
    end
    %legend(leg);
    % hold off;

    disp(['Best loss for act_func(',num2str(act_func),') was ', num2str(min), ' when alpha=',num2str(alpha(min_idx(2))),' and lambda=', num2str(lambda(min_idx(1))),'.']);
end