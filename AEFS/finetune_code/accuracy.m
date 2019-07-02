function acc = accuracy(Ws,bs,X,Y,activation_type)
    [~,Y_hat] = predict(Ws,bs,X,activation_type);
    dif = sum((Y-Y_hat).^2,2);
    acc = sum(dif==0)/size(Y,1);
end