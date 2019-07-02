function [Ws,bs] = finetune(X,Y,Ws,bs,ds,lambda,max_iter,loss_type,activation_type)
    if nargin < 8
        loss_type = 'softmax';
    end
    % init
    theta = Wb2theta(Ws,bs);
    % random sampling
    batch_size = size(X,1);
    idx = randsample(size(X,1),batch_size);
    xs = X(idx,:);
    ys = Y(idx,:);
    %  Use minFunc to minimize the function
    addpath ./GSAE/minFunc/
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost function.
    options.maxIter = max_iter;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';
    [opttheta, loss] = minFunc(@(theta) nnLoss(theta,ds,xs,ys,lambda,loss_type,activation_type), theta,options);
    [Ws,bs] = theta2Wb(opttheta,ds);
    % display
    train_acc = accuracy(Ws,bs,X,Y,activation_type);
    fprintf('loss : %f, train acc : %f\n',loss,train_acc);
end

% loss function
function [loss,grad] = nnLoss(theta,ds,xs,ys,lambda,loss_type,activation_type)
    if nargin < 6
        loss_type = 'softmax';
    end
    [Ws,bs] = theta2Wb(theta,ds);
    nl = size(Ws,2)+1;
    % forward propagation
    as{1} = xs;
    for i = 2:nl-1
        zs{i} = as{i-1}*Ws{i-1}+repmat(bs{i-1},size(as{i-1},1),1);
        as{i} = activation(zs{i}, activation_type);
    end
    zs{nl} = as{nl-1}*Ws{nl-1}+repmat(bs{nl-1},size(as{nl-1},1),1);
    if(strcmp(loss_type,'softmax'))
        as{nl} = softmax(zs{nl});
        deltas{nl} = -(ys-as{nl});
    elseif(strcmp(loss_type,'sq-error'))
        as{nl} = activation(zs{nl}, 'sigmoid');
        deltas{nl} = -(ys-as{nl}).*grad_activation(zs{nl}, 'sigmoid');
    end
    % compute delta of each layer
    for i = nl-1:-1:2
        deltas{i} = (deltas{i+1}*Ws{i}').*grad_activation(zs{i}, activation_type);
    end
    % compute partial derivative of each layer
    for i = nl-1:-1:1
        partial_Ws{i} = (as{i}'*deltas{i+1})/size(as{i},1)+lambda*Ws{i};
        partial_bs{i} = sum(deltas{i+1},1)/size(deltas{i+1},1);
    end
    % straighten
    grad = Wb2theta(partial_Ws,partial_bs);
    % compute loss
    W2sum = 0;
    for i = 1:size(Ws,2)
        W2sum = W2sum + sum(sum(Ws{i}.^2));
    end
    if(strcmp(loss_type,'softmax'))
        loss = -sum(sum(ys.*log(as{nl})))/size(ys,1)+lambda/2*W2sum;
    elseif(strcmp(loss_type,'sq-error'))    
        loss = sum(0.5*sum((ys-as{nl}).^2,2))/size(ys,1)+lambda/2*W2sum;
    end
end
% softmax
function [p] = softmax(z)
    p = exp(z);
    p = p./repmat(sum(p,2),1,size(p,2));
end
% % softmax delta
% function [delta] = softmax_delta(W,b,xs,ys)
%     p = exp(xs*W+repmat(b,size(xs,1),1));
%     p = p./sum(p);
%     delta = -(ys-p);
% end
% W/b to theta
function [theta] = Wb2theta(Ws,bs)
    theta = [];
    for i=1:size(Ws,2)
        theta = [theta;Ws{i}(:)];
    end
    for i=1:size(Ws,2)
        theta = [theta;bs{i}(:)];
    end
end
% theta to W/b
function [Ws,bs] = theta2Wb(theta,ds)
    left = 1;
    for i=1:size(ds,2)-1
        cur_number = ds{i}*ds{i+1};
        Ws{i} = reshape(theta(left:left+cur_number-1),ds{i},ds{i+1});
        left = left+cur_number;
    end
    for i=1:size(ds,2)-1
        cur_number = ds{i+1};
        bs{i} = reshape(theta(left:left+cur_number-1),1,ds{i+1});
        left = left+cur_number;
    end
end

% check the correctness of partial derivative
function [] = check_partial(layer,Ws,bs,lambda,xs,ys,partial_Ws,partial_bs)
    % W
    check_idx = randsample(numel(Ws{layer}),1);
    increment = 1e-4*zeros(size(Ws{layer}));
    increment(check_idx) = 1e-4;
    Wsplus = Ws; Wsplus{layer} = Ws{layer}+increment;
    Wsminus = Ws; Wsminus{layer} = Ws{layer}-increment;
    partial_val = (loss(Wsplus,bs,lambda,xs,ys)-loss(Wsminus,bs,lambda,xs,ys))./(2*increment);
    if(abs(partial_val(check_idx)-partial_Ws{layer}(check_idx))<1e-4)
        fprintf('partial derivative of W%d is correct!!!\n',layer);
    else
        fprintf(2,'partial derivative of W%d is wrong!!!\n',layer);
    end
    % b
    check_idx = randsample(numel(bs{layer}),1);
    increment = 1e-4*zeros(size(bs{layer}));
    increment(check_idx) = 1e-4;
    bsplus = bs; bsplus{layer} = bs{layer}+increment;
    bsminus = bs; bsminus{layer} = bs{layer}-increment;
    partial_val = (loss(Ws,bsplus,lambda,xs,ys)-loss(Ws,bsminus,lambda,xs,ys))./(2*increment);
    if(abs(partial_val(check_idx)-partial_bs{layer}(check_idx))<1e-4)
        fprintf('partial derivative of b%d is correct!!!\n',layer);
    else
        fprintf(2,'partial derivative of b%d is wrong!!!\n',layer);
    end
end
