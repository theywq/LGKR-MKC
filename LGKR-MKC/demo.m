close all; clear all; clc
warning off;
addpath(genpath('ClusteringMeasure'));
addpath(genpath('function'));
ResSavePath = 'Res/';
MaxResSavePath = 'maxRes/';
dataPath='../Datasets/';
datasetName = {'texas_Kmatrix', 'wisconsin_Kmatrix', 'AR10P_Kmatrix', 'PIE10P_Kmatrix', 'YALE_Kmatrix', 'Carcinom_173_11_Kmatrix', 'movement_libras_360_Kmatrix', 'caltech101_nTrain20_48_Kmatrix'};
for dataIndex = 2:length(datasetName) - (length(datasetName) - 2)
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    load(dataName, 'KH', 'Y');
    
    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);
    % Data Preparation
    tic;
    c = max(Y);
    V = size(KH, 3);
    N = size(KH, 1);
    KH = kcenter(KH);
    KH = knorm(KH);
    time1 = toc;    
    % parameters setting
    r1 = -5:2:5;
    r2 = -5:2:5;
    acc = zeros(length(r1), length(r2));
    nmi = zeros(length(r1), length(r2));
    purity = zeros(length(r1), length(r2));
    idx = 1;
    for r1Index = 1:length(r1)
        r1Temp = r1(r1Index);
        for r2Index = 1:length(r2)
            r2Temp = r2(r2Index);
            tic;
            % Main algorithm
            fprintf('Please wait a few minutes\n');
            disp(['Dataset: ', datasetName{dataIndex}, ...
                ', --r1--: ', num2str(r1Temp), ', --r2--: ', num2str(r2Temp)]);
            [F, G, Z] = LGKR(KH, c, 2.^r1Temp, 5.^r2Temp); 
            time2 = toc;
            tic;
            [res] = myNMIACC(real(F), Y, c);
            time3 = toc;
            Runtime(idx) = time1 + time2 + time3/20;
            disp(['runtime: ', num2str(Runtime(idx))]);
            idx = idx + 1;
            tempResBest(1, :) = res(1, :);
            tempResStd(1, :) = res(2, :);
            
            acc(r1Index, r2Index) = tempResBest(1, 7);
            nmi(r1Index, r2Index) = tempResBest(1, 4);
            purity(r1Index, r2Index) = tempResBest(1, 8);
            
            resFile = [ResSavePath datasetName{dataIndex}, '-ACC=', num2str(tempResBest(1, 7)), ...
                '-r1=', num2str(r1Temp), '-r2=', num2str(r2Temp), '.mat'];
            save(resFile, 'tempResBest', 'tempResStd', 'F');
            
            for tempIndex = 1:8
                if tempResBest(1, tempIndex) > ResBest(1, tempIndex)
                    if tempIndex == 7
                        newF = F;
                    end
                    ResBest(1, tempIndex) = tempResBest(1, tempIndex);
                    ResStd(1, tempIndex) = tempResStd(1, tempIndex);
                end
            end
        end
    end
    aRuntime = mean(Runtime);
    resFile2 = [MaxResSavePath datasetName{dataIndex}, '-ACC=', num2str(ResBest(1, 7)), '.mat'];
    save(resFile2, 'ResBest', 'ResStd', 'acc', 'nmi', 'purity', 'aRuntime', 'Y');
end

