function [nmi_mean,nmi_std,acc_mean,acc_std] = repeatkmeans(fea,class_num,gnd,repeat_times)
    nmis = zeros(repeat_times,1);
    accs = zeros(repeat_times,1);
    for i=1:repeat_times
      label = litekmeans(fea,class_num,'Replicates',1);
      % NMI and acc
      nmis(i) = nmi(gnd,label);
      bestLabel = bestMap(gnd,label);
      accs(i) = sum(bestLabel==gnd)/length(gnd);
    end
    % mean and std
    nmi_mean = mean(nmis);
    nmi_std = std(nmis);
    acc_mean = mean(accs);
    acc_std = std(accs);
end