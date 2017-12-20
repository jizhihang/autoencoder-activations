function [ w, v, loss ] = train_network( X, y, X_val, y_val, num_hidden, act_func, alpha, lambda, batch, epsilon, max_epoch, verbose )
    % Train Network: trains a 2-layer neural network
    % Parameters:
    %   x_train         - Input Vector (n x m)
    %   y_train         - Output Vector (n x p)
    %   x_val           - Input Validation Vector (n_val x m_val)
    %   y_val           - Output Validation Vector (n_val x p_val)
    %   num_hidden  - The number of hidden nodes in the network
    %   act_func    - The activation function desired:
    %         (1) Linear
    %         (2) Logistic (sigmoid)
    %         (3) TanH
    %         (4) ReLu
    %         (5) ELU
    %   alpha       - The step size
    %   lambda      - The l2 regularization term
    %   batch       - The batch size
    %   epsilon     - loss convergence criteria
    %   max_epoch   - The maximum number of epochs
    %   verbose     - If true the network prints status of training
    %
    % Returns:
    %   w           - The learned weights from input to hidden layer
    %   v           - The learned weights from hidden to output layer
    %   t           - The number of epochs
    
    if nargin<12, verbose = false; end
    if nargin<11, max_epoch = 100; end
    if nargin<10, epsilon = 10; end
    if nargin<9, batch = 100; end
    if nargin<8, lambda = 0; end
    if nargin<7, alpha = 1e-2; end
    if nargin<6, act_func = 1; end
    if nargin<5, num_hidden = 784; end
    
    rng('default');
    
    [n,m] = size(X); % input layer dimensions
    [n_val,m_val] = size(X_val);
    t = 1;
    
    err = Inf;
    % Generate normally distributed weight matrices with pseudorandom numbers.
    w = randn(m+1, num_hidden); % weights for inputs-hidden
    v = randn(num_hidden+1, m); % weights for hidden-outputs
    
    loss(t) = norm(act([ones(n_val,1) act([ones(n_val,1) X_val]*w, act_func)]*v, act_func)-y_val);

    %% Train Autoencoder
    while( (t <= max_epoch) && (err > epsilon) )

        % Pick random images
        img_idx = randperm(n);
        idx_lo = 1;
        
        reverseStr = '';
        for i = 1:ceil(n/batch)
            if(verbose)
                msg = sprintf('Epoch %d Progress: %3.1f', t+1, 100 * i / ceil(n/batch));
                fprintf([reverseStr, msg]);
                reverseStr = repmat(sprintf('\b'), 1, length(msg));
            end
            
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
        
        if(verbose), fprintf('\nL2 Validation Loss After Epoch %d Is %1.2e\n',t-1,loss(t)); end
        
        if isnan(loss(t)), return; end%loss = cat(2, loss, zeros(max_epoch - t,1)'); return; end
        if((loss(t)-loss(t-1))<epsilon), return; end;
        
    end
end