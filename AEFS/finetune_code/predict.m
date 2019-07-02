function [h,Y_hat] = predict(Ws,bs,X,activation_type)
    nl = size(Ws,2)+1;
    % forward propagation
    as{1} = X;
    for i = 2:nl-1
        zs{i} = as{i-1}*Ws{i-1}+repmat(bs{i-1},size(as{i-1},1),1);
        as{i} = activation(zs{i},activation_type);
    end
    zs{nl} = as{nl-1}*Ws{nl-1}+repmat(bs{nl-1},size(as{nl-1},1),1);
    h = softmax(zs{nl});
    Y_hat = zeros(size(h));
    [~,idx] = max(h,[],2);
    for i = 1:length(idx)
        Y_hat(i,idx(i)) = 1;
    end
end

% softmax
function [p] = softmax(z)
    p = exp(z);
    p = p./repmat(sum(p,2),1,size(p,2));
end

