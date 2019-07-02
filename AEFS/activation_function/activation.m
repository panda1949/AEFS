function [ a ] = activation( z,type )
    % default: sigmoid
    if nargin == 1
        type = 'sigmoid';
    end 
    % sigmoid function
    if strcmp(type, 'sigmoid')
        a = 1./(1+exp(-z));
    % tanh function
    elseif strcmp(type, 'tanh')
        a = tanh(z);
    % relu function
    elseif strcmp(type, 'relu')
        a = max(0, z);
    elseif strcmp(type, 'self')
        a = z;
    end
end

