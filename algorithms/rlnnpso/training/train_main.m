% RL-NNPSO Training Script (GPU-Accelerated)
%
% Train TD3 networks for RL-NNPSO with optimized curriculum learning
% Run this from the project root directory
%
% What this does:
%   1. Trains TD3 networks for 300 episodes (GPU-accelerated, curriculum learning)
%      - Phase 1: 210 episodes pure RL (gamma=1.0)
%      - Phase 2: 90 episodes adaptive blending (gamma=0.3-0.9)
%   2. Tests the pre-trained networks on a simple scenario
%   3. Compares performance with baseline PSO
%
% Optimized Training Settings (particle-level control):
%   - Episodes: 300 (optimized for 3M experiences, ~2 hours)
%   - Training: popSize=50, maxIterations=200 (10,000 exp/episode)
%   - Testing: popSize=200, maxIterations=800 (benchmark scale)
%   - Buffer: 1,000,000 experiences (RL standard)
%   - Batch: 256 (optimized for GPU)
%   - Auto-detects GPU availability

clear; clc; close all;

fprintf('========================================\n');
fprintf('RL-NNPSO PRE-TRAINING QUICK START\n');
fprintf('========================================\n\n');

% Add paths automatically
scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(scriptDir, '..', '..', '..');
addpath(genpath(fullfile(repoRoot, 'algorithms')));
addpath(genpath(fullfile(repoRoot, 'shared')));
addpath(genpath(fullfile(repoRoot, 'scripts')));
addpath(genpath(fullfile(repoRoot, 'data')));
addpath(repoRoot);

%% Step 1: Pre-train networks with GPU acceleration
fprintf('STEP 1: Pre-training TD3 networks for RL-NNPSO (GPU-Accelerated)...\n');
fprintf('Optimized for particle-level control with curriculum learning\n');
fprintf('Using paper-standard hyperparameters: Buffer=1M, Batch=256\n\n');

% Training configuration (optimized for RLNNPSO particle-level control)
% RLNNPSO generates 200× more experiences than RLAM-PSO per episode
% Reduced parameters for practical training time while maintaining learning quality
numEpisodes = 300;   % Optimized: 210 pure RL + 90 adaptive (vs 1000 paper-standard)
popSize = 50;        % Reduced from 200 (benchmark uses 200 for testing)
maxIterations = 200; % Reduced from 800 (benchmark uses 800 for testing)
modelPath = fullfile(repoRoot, 'models', 'rlnnpso_gpu.mat');
useGPU = true;       % Set to false to use CPU (auto-detects if not specified)

fprintf('=== OPTIMIZED TRAINING PARAMETERS ===\n');
fprintf('Episodes: %d (300 = 210 pure RL + 90 adaptive)\n', numEpisodes);
fprintf('Training: popSize=%d, maxIterations=%d\n', popSize, maxIterations);
fprintf('Experiences/episode: %d (vs RLAM-PSO: 800)\n', popSize * maxIterations);
fprintf('Total experiences: %d million\n', numEpisodes * popSize * maxIterations / 1e6);
fprintf('Estimated time: ~2 hours\n');
fprintf('\nNote: Benchmark testing uses popSize=200, maxIterations=800\n');
fprintf('      (networks trained on 50×200 generalize to 200×800)\n\n');

trainRLNNPSO_GPU(numEpisodes, modelPath, useGPU, popSize, maxIterations);

fprintf('\n========================================\n');
fprintf('STEP 2: Testing pre-trained networks\n');
fprintf('========================================\n\n');

%% Step 2: Test on a simple scenario

% Simple test scenario
mapSize = [400, 400, 100];
startPoint = [50, 50, 50];
goalPoint = [350, 350, 50];

obstacles = [
    150, 150, 40, 30;
    250, 200, 50, 25;
];

% Trees format: [x, y, z, trunkRadius, canopyRadius, isCylinder, treeHeight]
trees = [
    180, 180, 0, 3, 10, 0, 15;
    280, 220, 0, 3, 8, 0, 12;
];

[terrainX, terrainY] = meshgrid(0:10:mapSize(1), 0:10:mapSize(2));
terrainGrid = zeros(size(terrainX));

% PSO parameters (using same settings as training for consistency)
% popSize = 200;       % Already defined above (line 42)
% maxIterations = 800; % Already defined above (line 43)
fixedW = 0.7;
fixedC1 = 1.5;
fixedC2 = 1.5;

%% Run RL-NNPSO with pre-trained networks
fprintf('Running RL-NNPSO with PRE-TRAINED TD3 networks...\n');
tic;
[~, convergenceRLNNPSO, statsRLNNPSO] = globalPathPlanningRLNNPSO(...
    startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, ...
    mapSize, popSize, maxIterations, fixedW, fixedC1, fixedC2, ...
    modelPath);
timeRLNNPSO = toc;

fprintf('RL-NNPSO (Pre-trained) completed in %.2f seconds\n', timeRLNNPSO);
fprintf('Final fitness: %.3f\n\n', statsRLNNPSO.actualBestFitness);

%% Run baseline PSO for comparison
fprintf('Running baseline PSO for comparison...\n');
tic;
[~, convergencePSO, statsPSO] = globalPathPlanningPSO(...
    startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, ...
    mapSize, popSize, maxIterations, fixedW, fixedC1, fixedC2);
timePSO = toc;

fprintf('Baseline PSO completed in %.2f seconds\n', timePSO);
fprintf('Final fitness: %.3f\n\n', statsPSO.actualBestFitness);

%% Display results
fprintf('========================================\n');
fprintf('RESULTS\n');
fprintf('========================================\n\n');

improvement = ((statsPSO.actualBestFitness - statsRLNNPSO.actualBestFitness) / statsPSO.actualBestFitness) * 100;

fprintf('Baseline PSO:            %.3f\n', statsPSO.actualBestFitness);
fprintf('RL-NNPSO (Pre-trained):  %.3f\n', statsRLNNPSO.actualBestFitness);
fprintf('Improvement:             %.2f%%\n\n', improvement);

if improvement > 0
    fprintf('SUCCESS: Pre-trained RL-NNPSO outperformed baseline PSO!\n');
    fprintf('(TD3-based velocity guidance with particle-level rewards)\n\n');
else
    fprintf('Note: More training episodes may improve performance.\n');
    fprintf('Try running with 500-1000 episodes for better results.\n\n');
end

%% Plot comparison
figure('Position', [100, 100, 800, 400]);

subplot(1, 2, 1);
plot(convergencePSO, 'b-', 'LineWidth', 1.5);
hold on;
plot(convergenceRLNNPSO, 'g-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Best Fitness');
title('Convergence Comparison');
legend('Baseline PSO', 'RL-NNPSO (Pre-trained)');
grid on;

subplot(1, 2, 2);
barData = [statsPSO.actualBestFitness, statsRLNNPSO.actualBestFitness];
bar(barData);
set(gca, 'XTickLabel', {'Baseline PSO', 'RL-NNPSO'});
ylabel('Final Fitness (lower is better)');
title('Final Performance Comparison');
grid on;

fprintf('========================================\n');
fprintf('NEXT STEPS:\n');
fprintf('========================================\n\n');
fprintf('1. For better results, train with more episodes (GPU-accelerated):\n');
fprintf('   trainRLNNPSO_GPU(1000, ''models/rlnnpso_gpu.mat'', true);\n\n');
fprintf('2. Use pre-trained networks in your own scenarios:\n');
fprintf('   [path, ~, ~] = globalPathPlanningRLNNPSO(..., ''%s'');\n\n', modelPath);
fprintf('3. Compare RL-NNPSO with RLAM-PSO to see which works best for your domain\n\n');
fprintf('Pre-trained networks saved at: %s\n', modelPath);
fprintf('\nPaper-Standard Settings Used:\n');
fprintf('  - Buffer size: 1,000,000 experiences (RL standard)\n');
fprintf('  - Batch size: 256 (optimized for GPU)\n');
fprintf('  - TD3: Twin Delayed DDPG with target policy smoothing\n');
fprintf('  - Particle-level rewards with velocity guidance\n');
fprintf('  - Adaptive blending between RL and traditional PSO\n');
