function [ w, v, loss ] = train_network( X, y, X_val, y_val, num_hidden, act_func, alpha, epsilon, lambda, batch, max_epoch )
    % Train Network: trains a 2-layer neural network
    % Parameters:
    %   X           - Input Vector (n x m)
    %   y           - Output Vector (n x p)
    %   num_hidden  - The number of hidden nodes in the network
    %   act_func    - The activation function desired:
    %         (1) Linear
    %         (2) Logistic (sigmoid)
    %         (3) TanH
    %         (4) ReLu
    %         (5) ELU
    %   alpha       - The step size
    %   epsilon     - The convergence criteria (l2-norm of v)
    %   lambda      - The l2 regularization term
    %   batch       - The batch size
    %   max_epoch   - The maximum number of epochs
    %
    % Returns:
    %   w           - The learned weights from input to hidden layer
    %   v           - The learned weights from hidden to output layer
    %   t           - The number of epochs
    
    if nargin<11, max_epoch = 10; end
    if nargin<10, batch = 100; end
    if nargin<9, lambda = 0; end
    if nargin<8, epsilon = 1e-4; end
    if nargin<7, alpha = 1e-2; end
    if nargin<6, act_func = 1; end
    if nargin<5, num_hidden = 784; end
    
    rng('default');
    
    [n,m] = size(X); % input layer dimensions
    [n_val,m_val] = size(X_val);
    t = 0;
    dist = Inf; % initialize to some value that clearly has not converged yet
    

    % Generate normally distributed weight matrices with pseudorandom numbers.
    w = randn(m+1, num_hidden); % weights for inputs-hidden
    v = randn(num_hidden+1, m); % weights for hidden-outputs


    %% Train Autoencoder
    while( (dist > epsilon) && (t < max_epoch) )

        % Pick random images
        img_idx = randperm(n);
        idx_lo = 1;
        
        reverseStr = '';
        for i = 1:ceil(n/batch)
            msg = sprintf('Epoch %d Progress: %3.1f', t+1, 100 * i / ceil(n/batch));
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
            
            idx_hi = idx_lo + batch - 1;
            if idx_hi > n, idx_hi = n; end
            len = idx_hi-idx_lo+1;

            x_batch =[ones(len,1) X(img_idx(idx_lo:idx_hi),:)];
            y_batch =y(img_idx(idx_lo:idx_hi),:);

            % Forward Propogation
            h_p =  x_batch * w;
            h = [ones(len,1) act(h_p, act_func)];
            o_p = h * v;
            o = act(o_p, act_func);

            % Backpropagation
            deltaoh = (o - y_batch).*deriv(o_p, act_func);
            v_new = v - alpha * h' * deltaoh - lambda*v;
            deltahi = deltaoh * v(2:end, :)' .* deriv(h_p, act_func);
            w_new = w - alpha * x_batch' * deltahi - lambda*w;

            % Update Error
            dist = norm(v_new - v);

            % Update weights
            w = w_new;
            v = v_new;
                        
            idx_lo = idx_lo + batch;
        end
        % Go to next epoch
        t = t + 1;
        loss(t) = norm(act([ones(n_val,1) act([ones(n_val,1) X_val]*w, act_func)]*v, act_func)-y_val);
        fprintf('\nL2 Validation Loss After Epoch %d Is %3.1e\n',t,loss(t))
    end
end