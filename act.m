function [ output ] = act( matrix, act_func )
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
        output = matrix;
    case 2
        output = logsig(matrix)%(1./(1+exp(-1.*matrix)));
    case 3
        output = ((2./(1+exp(-2.*matrix))) -1); 
    case 4
        output = max(0, matrix);
    case 5
        output = softmax(matrix);
    otherwise
        output = -1;
end
end

