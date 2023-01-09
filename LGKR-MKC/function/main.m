clear
clc
warning off;

for index = 7
DataName{1} = 'texas';
DataName{2} = 'cornell';
DataName{3} = 'washington';
DataName{4} = 'wisconsin';
DataName{5} = 'AR10P';
% DataName{6} = 'heart';
DataName{6} = 'PIE10P';
DataName{7} = 'YALE';
% DataName{9} = 'liver';
DataName{8} = 'ORL';
DataName{9} = 'proteinFold';
DataName{10} = 'flower17';
DataName{11} = 'caltech101_nTrain5_48';
DataName{12} = 'caltech101_nTrain10_48';
DataName{13} = 'caltech101_nTrain15_48';
DataName{14} = 'caltech101_nTrain20_48';
DataName{15} = 'caltech101_nTrain25_48';
DataName{16} = 'caltech101_nTrain30_48';
path = './';
addpath(genpath(path));
dataName = DataName{index}
load(['E:\mywork\mywork_mine\ML\Clustering\MKC\codes\Compared Methods\datasets\',dataName,'_Kmatrix'],'KH','Y');
% load([path,'datasets/',dataName,'_Kmatrix'],'KH','Y');
%% initialization
k = length(unique(Y)); %class_num
m = size(KH, 3); %kernel_num
n = size(KH, 1); %sample_num

KH = kcenter(KH);
KH = knorm(KH);

% lambda = 10.^[-10:10];
% beta = 10.^[10:15];
lambda = 2.^[-5:2:5];
beta = 10.^[-5:2:5];


acc = zeros(length(lambda),length(beta));
nmi = zeros(length(lambda),length(beta));
pur = zeros(length(lambda),length(beta));

i = 0;

for it = 1:length(lambda)
    for ir=1:length(beta)
        tic;
        [G, Z, obj] = mywork3(KH, k, lambda(it),beta(ir));
        savefig(['./','obj/',dataName,'-',int2str(it),'-',int2str(ir),'-obj.fig']);
%         S1 = (Z + Z') / 2;
%         D = diag(1 ./ sqrt(sum(S1)));
%         L =  D * S1 * D;
%         [H,~] = eigs(L, k, 'LA');
        K= (G+G')/2;
        opt.disp = 0;
        [H,~] = eigs(K,k,'LA',opt);
        H_normalized = H./ repmat(sqrt(sum(H.^2, 2)), 1,k);
        res= myNMIACC(H_normalized,Y,k);
        toc;
        acc(it, ir) = res(1);
        nmi(it, ir) = res(2);
        pur(it, ir) = res(3);
        i=i+1;
        fprintf('\nlambda: %d, beta: %d',lambda(it),beta(ir));
        fprintf('\nacc: %f, nmi: %f, pur: %f\n', res(1), res(2), res(3)); 
    end
    results=acc;
    ps=bar3(results);
    xlabel('\beta');ylabel('\lambda');zlabel('ACC');
    xticklabels(beta);
    yticklabels(lambda);
    title([dataName]);
    savefig(['./','ps/',dataName,'-ps.fig']);
end

res = [max(max(max(acc))), max(max(max(nmi))), max(max(max(pur)))];
save(['./','res/',dataName,'-res.mat'],'res');
save(['./','acc/',dataName,'-acc.mat'],'acc');
%res_path = 'D:\Work\work2015\jiyuan\code\res\';
%save([res_path,data_name,'_cluster_res_selparam1.mat'],'acc','nmi','pur');
end