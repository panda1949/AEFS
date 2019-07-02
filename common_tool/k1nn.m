function [pred_list] = k1nn(X,label_list)
    Mdl = KDTreeSearcher(X);
    [Idx,Dis] = knnsearch(Mdl,X,'k',2);
    pred_list = label_list(Idx(:,2));
end