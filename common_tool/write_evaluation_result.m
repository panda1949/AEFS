function [bestACC,bestNMI,best_knn_acc] = write_evaluation_result(fea,class_num,gnd,FeaIndexs,FeaNumCandi,fpath)
    fs = fopen(fpath, 'a+');
    bestNMI = 0; bestNMIidx = 0;
    bestACC = 0; bestACCidx = 0;
    best_knn_acc = 0; best_knn_idx=0;
    for i=1:length(FeaIndexs)
        fprintf(fs,[num2str(i),'th param result\r\n']);
        for j=1:length(FeaNumCandi)
            FeaIndex{j}=FeaIndexs{i}{j};
        end
        %Clustering using selected features
        for j = 1:length(FeaNumCandi)
          SelectFeaIdx = FeaIndex{j};
          feaNew = fea(:,SelectFeaIdx);
          % repeat 20 times
          [nmi_mean,nmi_std,acc_mean,acc_std] = repeatkmeans(feaNew,class_num,gnd,20);
          % NMI
          fprintf(fs,['Selected feature num: ',num2str(FeaNumCandi(j)),', Clustering MIhat: ',num2str(nmi_mean),'\n']);
          if(nmi_mean>bestNMI)
          	bestNMI = nmi_mean;
            bestNMI_std = nmi_std;
            bestNMIidx = j;
          end
          % ACC
          fprintf(fs,['Selected feature num: ',num2str(FeaNumCandi(j)),', Clustering ACC: ',num2str(acc_mean),'\n']);
          if(acc_mean>bestACC)
            bestACC = acc_mean;
            bestACC_std = acc_std;
            bestACCidx = j;
          end
          % knn
          pred_list = k1nn(feaNew,gnd);
          knn_acc = sum(pred_list==gnd)/length(gnd);
          fprintf(fs,['Selected feature num: ',num2str(size(feaNew,2)),', Classification ACC: ',num2str(knn_acc),'\r\n']);
          if(knn_acc>best_knn_acc)
              best_knn_acc = knn_acc;
              best_knn_idx = j;
          end
        end
    end
    fprintf(fs,['\r\n Best: Selected feature num: ',num2str(FeaNumCandi(bestNMIidx)),', Clustering MIhat: ',num2str(bestNMI),...
        ', std: ',num2str(bestNMI_std),'\r\n']);
    fprintf(fs,['\r\n Best: Selected feature num: ',num2str(FeaNumCandi(bestACCidx)),', Clustering ACC: ',num2str(bestACC),...
        ', std: ',num2str(bestACC_std),'\r\n']);
    fprintf(fs,['\r\n Best: Selected feature num: ',num2str(FeaNumCandi(best_knn_idx)),', Classification ACC: ',num2str(best_knn_acc),'\r\n']);
    fclose(fs);
end