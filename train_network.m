function [ w, v, t ] = train_network( X, y, num_hidden, act_func, alpha, epsilon, batch, max_epoch )
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
    %   batch       - The batch size
    %   max_epoch   - The maximum number of epochs
    %
    % Returns:
    %   w           - The learned weights from input to hidden layer
    %   v           - The learned weights from hidden to output layer
    %   t           - The number of epochs
    
    if nargin<8, max_epoch = 10; end
    if nargin<7, batch = 100; end
    if nargin<6, epsilon = 1e-4; end
    if nargin<5, alpha = 1e-2; end
    if nargin<4, act_func = 1; end
    if nargin<3, num_hidden = 784; end

    [m, n] = size(X); % input layer dimensions
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
        

        for i = 1:ceil(n/batch)        
            idx_hi = idx_lo + batch - 1;
            if idx_hi > n, idx_hi = n; end
            len = idx_hi-idx_lo+1;

            x_batch =[ones(len,1) X(:,img_idx(idx_lo:idx_hi))'];
            y_batch =y(:,img_idx(idx_lo:idx_hi))';

            % Forward Propogation
            h_p =  x_batch * w;
            h = [ones(len,1) act(h_p, act_func)];
            o_p = h * v;
            o = act(o_p, act_func);

            % Backpropagation
            deltaoh = (o - y_batch).*deriv(o, act_func);
            vbp = v - alpha * h' * deltaoh;
            deltahi = deltaoh * v(2:end, :)' .* deriv(h(:,2:end), act_func);
            wbp = w - alpha * x_batch' * deltahi;

            % Update Error
            dist = norm(vbp - v);

            % Update weights
            w = wbp;
            v = vbp;
            
            idx_lo = idx_lo + batch;
        end

            % Go to next epoch
            t = t + 1; 
    end
end