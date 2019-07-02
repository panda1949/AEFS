function [Ws,bs] = pretrain(X,ds,lambda,beta,denoising,noise_level,max_iter,activation_type)
    nl = size(ds,2);
    % init
    amp = 5e-3;
    for i=1:nl-1
        Ws{i} = amp*randn(ds{i},ds{i+1});
        bs{i} = amp*randn(1,ds{i+1});
    end
    % max_iter>0, using sparse autoencoder to pretrain
    if(max_iter>0)
        % sparse autoencoder pretrain for each layer, except the last output layer
        A = X;
        for i = 1:nl-2
            if denoising==1
                noise_list = [1e-2,1e-1,1,3];
                noise_A = A+noise_list(noise_level)*randn(size(X));
            elseif denoising==2
                noise_list = [0.2,0.4,0.6,0.8];
                zero_idx = randsample(size(A,2),floor(size(A,2)*noise_list(noise_level)));
                noise_A = A;
                noise_A(:,zero_idx)=0;
            else
                noise_A = A;
            end
            [Ws{i},bs{i}] = sparse_autoencoder(noise_A,A,ds{i+1},lambda,beta,max_iter,activation_type);
            % hidden layer 
            Z = A*Ws{i}+repmat(bs{i},size(A,1),1);
            A = activation(Z,activation_type);
        end
    end
end
