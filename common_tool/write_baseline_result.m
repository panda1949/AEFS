function [acc_mean,nmi_mean,knn_acc] = write_baseline_result(fea,class_num,gnd,fpath)
    fs = fopen(fpath, 'a+');
    fprintf(fs, ['\r\n***** tuning parameters for ',fpath,'*****\r\n']);
    % cluster
    [nmi_mean,nmi_std,acc_mean,acc_std] = repeatkmeans(fea,class_num,gnd,20);
    fprintf(fs,['Clustering using all the ',num2str(size(fea,2)),' features. Clustering MIhat: ',num2str(nmi_mean),...
        ', std: ',num2str(nmi_std),'\r\n']);
    fprintf(fs,['Selected feature num: ',num2str(size(fea,2)),', Clustering ACC: ',num2str(acc_mean),...
        ', std: ',num2str(acc_std),'\r\n']);
    % knn classification
    pred_list = k1nn(fea,gnd);
    knn_acc = sum(pred_list==gnd)/length(gnd);
    fprintf(fs,['Selected feature num: ',num2str(size(fea,2)),', Classification ACC: ',num2str(knn_acc),'\r\n']);
    fclose(fs);
end