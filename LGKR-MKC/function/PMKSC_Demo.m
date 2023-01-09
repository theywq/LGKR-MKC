clear
clc
warning off;
for index=11
DataName{1} = 'AR10P';
DataName{2} = 'YALE';
DataName{3} = 'proteinFold';
DataName{4} = 'flower17';
DataName{5} = 'caltech101_nTrain5_48';
DataName{6} = 'caltech101_nTrain10_48';
DataName{7} = 'caltech101_nTrain15_48';
DataName{8} = 'mfeat';
DataName{9} = 'caltech101_nTrain20_48';
DataName{10} = 'caltech101_nTrain25_48';
DataName{11} = 'flower102';
DataName{12} = 'caltech101_nTrain30_48';


lambdaset = 2.^[-2:1:2];
%betaset = 2.^[-2:1:2];
%gammaset = 10.^[-5:2:5];

DataHyperParam{10} = [
    6,5,5;
    ];

count = 1;
for i=1:length(lambdaset)
    %for j=1:length(betaset)
        %for k=1:length(gammaset)
            allsort(count,:) = [i];
            count = count + 1;
        %end
    %end
end

DataHyperParam{index} = allsort;

;

path = './';
addpath(genpath(path));
k=0;

    k=k+1;
    dataName = DataName{index} %%% flower17; flower102; proteinFold,caltech101_mit,UCI_DIGIT,ccv
    dataHyperParam = DataHyperParam{index};
    load([path,'datasets/',dataName,'_Kmatrix'],'KH','Y');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    numclass = length(unique(Y));
    numker = size(KH,3);
    num = size(KH,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    KH = kcenter(KH);
    KH = knorm(KH);
    K0 = zeros(num,num); 
    qnorm = 2;
    opt.disp = 0;
    tic
    
    %accval9=zeros(length(lambdaset),length(betaset),length(gammaset));
    %nmival9=zeros(length(lambdaset),length(betaset),length(gammaset));
    %purval9=zeros(length(lambdaset),length(betaset),length(gammaset));
    
    for param=1:size(dataHyperParam,1)
        it = dataHyperParam(param,1);
        ij = dataHyperParam(param,2);
        %iq = dataHyperParam(param,3);
        %disp(['p1=',num2str(lambdaset(it)), '  p2=',num2str(betaset(ij)), '  p3=',num2str(gammaset(iq))]);
        disp(['p1=',num2str(lambdaset(it)), '  p2=',num2str(betaset(ij))]);
        tic;
        
        [H_normalized9, iter, obj] = pmksclustering(KH,numclass,lambdaset(it),betaset(ij),Y,numclass);
        %save(['./',dataName,'-obj.mat'],'obj');
        savefig(['./','obj/',dataName,'-',int2str(it),'-',int2str(ij),'-obj.fig']);
        res9 = myNMIACCwithmean(H_normalized9,Y,numclass);
        time(k)=toc;
        %
        accval9(it,ij) = res9(1,1);      
        nmival9(it,ij)= res9(2,1);      
        purval9(it,ij) = res9(3,1);
    end
    results=accval9;
    ps=bar3(results);
    xlabel('\beta');ylabel('\lambda');zlabel('ACC');
    xticklabels(betaset);
    yticklabels(lambdaset);
    title([dataName]);
    savefig(['./','ps/',dataName,'-ps.fig']);
    res = [max(max(max(accval9))); max(max(max(nmival9)));max(max(max(purval9)))];
    save(['./','res/',dataName,'-res.mat'],'res');
    save(['./',dataName,'-acc.mat'],'accval9');
end


