function [ grad ] = grad_activation( z,type )
    % default: sigmoid
    if nargin == 1
        type = 'sigmoid';
    end 
    % sigmoid function
    if strcmp(type, 'sigmoid')
        e_z = exp(-z);
        grad = e_z ./ ((1 + e_z).^2); 
    % tanh function
    elseif strcmp(type, 'tanh')
        grad = 1-tanh(z).^2;
    % relu function
    elseif strcmp(type, 'relu')
        grad = 1.0*(z>=0);
    elseif strcmp(type, 'self')
        grad = 1.0;
    end
end

