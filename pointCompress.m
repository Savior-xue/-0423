% 两点压缩感知
% by John Chan
% March 2025

% Modified by YSun, Mar. 31, 2025

% Modified by Ruixue Sun, Apr. 2,2025

close all;
clear all;
clc;
rngsnSampler = 'default'; % 随机数种子字符串
rng(rngsnSampler);        % 固定随机种子

%% 基础参数设置
c = 3e8;                  % 光速 (m/s)
fs = 2.4e9;               % 基带频率 (2.4GHz)
fc = 6e9;                 % 采样率 (6GHz)
snr = -20 : 10 : 20;      % 信噪比范围 (dB)
nSample = 2^8;            % 采样点数
N = 500;                  % 每组信噪比下的样本数
senLen = 300;             % 传感器位置范围 (m)
unkLen = 300;             % 未知信号源位置范围 (m)

%% 传感器配置
s = 2 * senLen * rand(2, 2) - senLen; % 传感器坐标 (2x2矩阵)
r = norm(s(1, :) - s(2, :));         % 基线长度 (m)
rd = -r:1:r;                       % 时延等效距离差范围 (m)

%% 数据集初始化
savePath = 'D:\MatLab';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end
if exist(fullfile(savePath, 'Dataset.mat'), 'file')
    delete(fullfile(savePath, 'Dataset.mat'));  % 清除旧文件
end

% 计算总样本数并预分配完整空间
totalSamples = length(snr) * length(rd) * N;
outputFile = matfile(fullfile(savePath, 'Dataset.mat'), 'Writable', true);
outputFile.Data(totalSamples, nSample*2) = single(0);  
outputFile.Label(totalSamples, 1) = single(0);  
disp('Simulation is running...');

%% 主处理循环
dataBuffer = single(zeros(totalSamples, nSample*2));  % 临时缓冲区
labelBuffer = single(zeros(totalSamples, 1)); 
sampleCounter = 1; % 独立样本计数器
for nseIndex = 1 : length(snr)
    fprintf('SNR = %d dB(%d / %d)...\n', snr(nseIndex), nseIndex, length(snr));
    
    % 噪声功率计算 (信号功率已归一化为1)
    nsePwr = 10^(-snr(nseIndex)/10);
    noiseBatch = sqrt(nsePwr/2) * single(randn(nSample, N, 2) + 1i*randn(nSample, N, 2));

    for nSrc = 1 : length(rd)
        fprintf('nSrc = %d',nSrc);
        % 时延计算 (秒)
        tau = single(rd(nSrc) / c);
        
        % 生成信号批次 (复数，功率归一化为1)
        x1Batch = (randn(nSample, N, 'single') + 1i*randn(nSample, N, 'single')) / sqrt(2);
        x2Batch = delayseq(x1Batch, tau);
        
        % 并行处理数据
        tempBuffer = single(zeros(N, nSample*2));
        parfor ndataIndex = 1:N
            % 添加噪声
            x1_nse = x1Batch(:, ndataIndex) + noiseBatch(:, ndataIndex, 1);
            x2_nse = x2Batch(:, ndataIndex) + noiseBatch(:, ndataIndex, 2);
            
            % 合并数据并转置为行向量
            tempBuffer(ndataIndex, :) = [x1_nse; x2_nse].';
        end
        
        % 直接写入预分配位置
        dataBuffer(sampleCounter:sampleCounter+N-1, :) = tempBuffer;
        labelBuffer(sampleCounter:sampleCounter+N-1) = repmat(tau*c, N, 1);
        sampleCounter = sampleCounter + N;
        
        % 释放内存
        clear x1Batch x2Batch tempBuffer;
    end
end
%% 写入文件
outputFile.Data = dataBuffer;
outputFile.Label = labelBuffer;
%% 清理资源
delete(gcp('nocreate'));
disp('Optimized simulation completed!');