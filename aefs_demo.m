clear;clc;close all;
%% demo of paper:
% Kai Han, Yunhe Wang, Chao Zhang, Chao Li, Chao Xu
% AutoEncoder Inspired Unsupervised Feature Selection
% ICASSP 2018
% 
% code author: Kai Han
% contact: kaihana@163.com
%
%% add path
addpath('./AEFS/activation_function');
addpath('./AEFS/pretrain_code');
addpath('./AEFS/finetune_code');
addpath('./AEFS/minFunc');
addpath('./common_tool');
rng(1);

%% load data and init
% dataset mat file should include fea, gnd
dataset = 'warpPIE10P';
load(['./data/',dataset,'.mat']);
fea = normalizemeanstd(fea);
gnd = gnd;
class_num = length(unique(gnd));

%% baseline: using all the features
fpath = ['aefs_result_',dataset,'.txt'];
write_baseline_result(fea,class_num,gnd,fpath);

%% Unsupervised feature selection using AEFS
% parameters
hidden_size = 256;
lambda = 3e-2; % weight decay parameter 
beta = 3e-1; % weight of sparsity penalty term  
pretrain_max_iter = 100;
activation_type = 'sigmoid';

% train
d1 = size(fea,2);
d2 = hidden_size;
d3 = length(unique(gnd));
ds = {d1,d2,d3};
[Ws,bs] = pretrain(fea,ds,lambda,beta,0,0,pretrain_max_iter,activation_type);

% get sorted weight of input
W1 = Ws{1};
row_norm = sum(W1.*W1,2);
[sorted_norm,sorted_idx] = sort(row_norm,1,'descend');
% sorted_idx is the index of selected features

%% evaluation
sorted_idxs{1}=sorted_idx;
FeaNumCandi = [50:50:300];
for i=1:length(sorted_idxs)
    for j=1:length(FeaNumCandi)
        FeaIndex{j}=sorted_idxs{i}(1:FeaNumCandi(j));
    end
    FeaIndexs{i} = FeaIndex;
end
write_evaluation_result(fea,class_num,gnd,FeaIndexs,FeaNumCandi,fpath);
