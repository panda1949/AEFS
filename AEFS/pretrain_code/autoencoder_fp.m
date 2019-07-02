function [h,Y_hat] = autoencoder_fp(Ws,bs,X,activation_type,last_activation_type)
    nl = size(Ws,2)+1;
    % forward propagation
    as{1} = X;
    for i = 2:nl-1
        zs{i} = as{i-1}*Ws{i-1}+repmat(bs{i-1},size(as{i-1},1),1);
        as{i} = activation(zs{i},activation_type);
    end
    zs{nl} = as{nl-1}*Ws{nl-1}+repmat(bs{nl-1},size(as{nl-1},1),1);
    h = activation(zs{nl},last_activation_type);
end