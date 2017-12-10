function [ output ] = deriv( matrix, act_func )
% ACT - returns a matrix after applying a specified activation function to
% all of its elements. If invalid `act` specified, returns -1. 
%   Activation functions: 
%     (1) Linear
%     (2) Logistic (sigmoid)
%     (3) TanH
%     (4) ReLu
%     (5) Softmax

switch act_func
    case 1
        output = 1;
    case 2
        output = (act(matrix, 2) .* (1 - act(matrix, 2)));
    case 3
        output = (1 - (act(matrix, 3)).^2);
    case 4
        output = matrix >= 0;
    case 5
        output = -1; % TODO: Implement softmax derivative
    otherwise
        output = -1;
end
end

