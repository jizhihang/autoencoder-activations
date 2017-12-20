%% CS 532 Final Project -- Rita Roloff, Justin Essert, Aaron Levin
close all; clear all;
%% Load data
train_images = loadMNISTImages('MNIST/train-images-idx3-ubyte');
test_images = loadMNISTImages('MNIST/t10k-images-idx3-ubyte');

% Training and testing images are known to be 28x28 from the dataset
% specifications, but, the manner in which we are loading them puts them
% into a neural-network-friendly 2D format of: 
%   784x60000 (training images)
%   60000x1   (training labels)

% Take 10% of training set and use it for validation
x = train_images';
indices = crossvalind('Kfold', ones(1, size(x, 1)), 10);
x_train = x(~(indices == 1),:);
x_val = x(indices == 1,:);


%% Setup hyperparameters

batch = 1000;           % Batch size
max_epoch = 25;         % Number of training iterations to run
epsilon = .1;
act_func_max = 5;       % max activation function codes
act2str = ["Linear", "Sigmoid","Tanh","ReLU","ELU"];

%% Find Alpha and Lambda values
num_hidden_init = 784;   % Number of hidden nodes for finding alpha and lambda

get_new_vals = true;
if(get_new_vals)
    [alpha, lambda] = find_params(x_train,x_train,x_val,x_val,num_hidden_init,batch,epsilon,max_epoch);
    save('alpha_lambda.mat','alpha','lambda')
else
    load('alpha_lambda.mat')
end

%% Train Network
num_hidden_support = [50, 20]; % Test num_hidden values on log scale
losses = zeros(size(act_func_max,2),size(num_hidden_support,2));

figure(1);
for act_func=1:act_func_max
    
    disp("Starting Activation Function: "+act2str(act_func));
    
    subplot(3,2,act_func)
    title("Activation Function: "+act2str(act_func));
    clear leg; hold on;
    
    for num_hidden=1:size(num_hidden_support,2)
        [ w, v, loss ] = train_network(x_train, x_train, x_val, x_val, num_hidden_support(num_hidden), act_func, alpha(act_func),lambda(act_func), batch, epsilon, max_epoch, true );
        t = size(loss,2);
        losses(act_func,num_hidden) = loss(t);
        % Handle case when weights go to Inf or loss converged
        if(t~=max_epoch+1)
            loss(t+1:max_epoch+1) = loss(t);
        end
        plot(0:max_epoch,loss);
        leg(num_hidden) = "num hidden = "+num2str(num_hidden_support(num_hidden));
    end
    legend(leg); hold off;

end