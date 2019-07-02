function [W1,b1] = sparse_autoencoder(X,Y,hidden_size,lambda,beta,max_iter,activation_type)
    d1 = size(X,2);
    d2 = hidden_size;
    d3 = size(Y,2);
    % init
    amp = 5e-3;
    Ws{1} = amp*randn(d1,d2);
    bs{1} = amp*randn(1,d2);
    Ws{2} = amp*randn(d2,d3);
    bs{2} = amp*randn(1,d3);
    theta = [Ws{1}(:);Ws{2}(:);bs{1}(:);bs{2}(:)];
    % random sampling
    batch_size = floor(size(X,1));
    idx = randsample(size(X,1),batch_size);
    xs = X(idx,:);
    ys = Y(idx,:);
    %  Use minFunc to minimize the function
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                              % function. Generally, for minFunc to work, you
                              % need a function pointer with two outputs: the
                              % function value and the gradient. In our problem,
                              % sparseAutoencoderCost.m satisfies this.
    options.maxIter = max_iter;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';
    [opttheta, loss] = minFunc(@(theta) sparseAutoencoderLoss(theta,d1,d2,d3,xs,ys,lambda,beta,activation_type), theta,options);
    Ws{1} = reshape(opttheta(1:d1*d2),d1,d2);
    Ws{2} = reshape(opttheta(d1*d2+1:d1*d2+d2*d3),d2,d3);
    bs{1} = reshape(opttheta(d1*d2+d2*d3+1:d1*d2+d2*d3+d2),1,d2);
    bs{2} = reshape(opttheta(d1*d2+d2*d3+d2+1:end),1,d3);
    W1 = Ws{1};
    b1 = bs{1};
    % display
    fprintf('loss : %f\n',loss);
end
% loss function
function [loss,grad] = sparseAutoencoderLoss(theta,d1,d2,d3,xs,ys,lambda,beta,activation_type)
    Ws{1} = reshape(theta(1:d1*d2),d1,d2);
    Ws{2} = reshape(theta(d1*d2+1:d1*d2+d2*d3),d2,d3);
    bs{1} = reshape(theta(d1*d2+d2*d3+1:d1*d2+d2*d3+d2),1,d2);
    bs{2} = reshape(theta(d1*d2+d2*d3+d2+1:end),1,d3);
    % forward propagation
    z2 = xs*Ws{1}+repmat(bs{1},size(xs,1),1);
    a2 = activation(z2, activation_type);
    z3 = a2*Ws{2}+repmat(bs{2},size(a2,1),1);
    h = activation(z3,'self');
    % compute delta of each layer
    delta3 = -(ys-h).*grad_activation(z3, 'self');
    delta2 = (delta3*Ws{2}').*grad_activation(z2, activation_type); 
    % compute partial derivative of each layer
    partial_Ws{2} = (a2'*delta3)/size(a2,1)+lambda*Ws{2};
    partial_bs{2} = sum(delta3,1)/size(delta3,1);
    partial_Ws{1} = (xs'*delta2)/size(xs,1)+lambda*Ws{1}+beta*repmat(1./sqrt(sum(Ws{1}.^2,2)),1,size(Ws{1},2)).*Ws{1};
    partial_bs{1} = sum(delta2,1)/size(delta2,1);
    % straighten
    grad = [partial_Ws{1}(:);partial_Ws{2}(:);partial_bs{1}(:);partial_bs{2}(:)];
    % compute loss
    sqLoss = sum(0.5*sum((ys-h).^2,2))/size(ys,1);
    W2sum = 0;
    for i = 1:size(Ws,2)
        W2sum = W2sum + sum(sum(Ws{i}.^2));
    end
    groupSparse = sum(sqrt(sum(Ws{1}.^2,2)));
    loss = sqLoss+lambda/2*W2sum+beta*groupSparse;
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
