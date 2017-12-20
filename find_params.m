function [alpha, lambda] = find_params(x_train,y_train,x_val,y_val,num_hidden,batch,epsilon,max_epoch)
    % Find Params: Finds the best alpha and lambda variables for each
    % activation function.
    % Parameters:
    %   x_train         - Input Vector (n x m)
    %   y_train         - Output Vector (n x p)
    %   x_val           - Input Validation Vector (n_val x m_val)
    %   y_val           - Output Validation Vector (n_val x p_val)
    %   num_hidden      - The number of hidden nodes in the network
    %   batch           - batch size
    %   epsilon         - loss convergence criteria
    %   max_epoch       - The maximum number of epochs
    %
    % Returns:
    %   alpha       - Array of best alpha values for each activation func
    %   lambda      - Array of best lambda values for each activation func
    
    if nargin<8, max_epoch = 100; end
    if nargin<7, epsilon = 10; end
    if nargin<6, batch = 100; end
    if nargin<5, num_hidden = 784; end
    
    act2str = ["Linear", "Sigmoid","Tanh","ReLU","ELU"];

    lambda_support = 10.^(2:-2:-4);
    alpha_support = 10.^[1:-1:-4  -6:-2:-10];
    
    for act_func=1:5
        min = Inf;
        min_idx = [0  0];
        
        disp("Starting Activation Function: "+act2str(act_func));

        for lam=1:size(lambda_support,2)
            
            for alph=1:size(alpha_support, 2)
                disp(['lambda = ',num2str(lambda_support(lam)),', alpha = ', num2str(alpha_support(alph))])
                [ w, v, loss ] = train_network(x_train, y_train, x_val, y_val, num_hidden, act_func, alpha_support(alph),lambda_support(lam), batch, epsilon, max_epoch, false );
                t = size(loss,2);
                if loss(t) < min, min = loss(t); min_idx = [lam alph]; end
            end
        end
        
        % Save and display best vars
        alpha(act_func) = alpha_support(min_idx(2));
        lambda(act_func) = lambda_support(min_idx(1));
        disp("Best loss for "+act2str(act_func)+" was "+ num2str(min)+" when alpha="+num2str(alpha(act_func))+" and lambda="+ num2str(lambda(act_func))+".");
    end
end