% =========================================================================
% RLAMPSO UAV PATH PLANNING TRAINING
% GPU-OPTIMIZED SEQUENTIAL TRAINING
% =========================================================================
% Simplified training script for UAV path planning only
%
% CONFIGURATION: All parameters are defined in RLAMPSO_Config.m
%   - Change RLAMPSO_MODE below to switch configurations
%   - Modify RLAMPSO_Config.m to change specific parameters
%
% TRAINING MODE: GPU-Optimized Sequential (NOT Parallel)
%   - Uses single GPU for stable dlnetwork training
%   - Larger batch sizes (512-1024) for GPU efficiency
%   - Faster and more stable than parallel CPU training
%
% STABILITY IMPROVEMENTS (2025-10-27):
%   - TAU = 0.125 (aggressive target network synchronization)
%   - Learning rates: Actor 1e-7, Critic 5e-6 (ultra-conservative)
%   - No gradient clipping (let Adam optimizer handle it)
%   - No learning rate warmup (not needed with small LR)
%   - Logarithmic reward scaling (handles large dynamic range)
%
% Output:
%   - uav_agent.mat (trained agent - deployment ready)
%   - training_*.log (complete training log)
%
% =========================================================================

clear; clc; close all;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     RLAMPSO UAV PATH PLANNING TRAINING (GPU-Optimized)       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% =========================================================================
%  CONFIGURATION
%% =========================================================================

% ===== RLAMPSO MODE =====
% Choose: 'baseline', 'advanced', 'td3_only', 'transformer_only', or 'fast'
% ALL PARAMETERS (episodes, workers, iterations, popSize, etc.)
% are loaded from RLAMPSO_Config(RLAMPSO_MODE)
RLAMPSO_MODE = 'baseline';

% ===== SAVE PATHS =====
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

%% =========================================================================
%  SETUP
%% =========================================================================

fprintf('Setting up paths and logging...\n');

% Setup MATLAB implementation paths
% Get script directory (training folder) and go up to matlabimplementation root
scriptDir = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(scriptDir);

% Temporarily add matlabRoot to path to access setupPaths
addpath(matlabRoot);

% Call setup function
setupPaths();

% Create directories in the training folder where the script runs
pretrainedDir = fullfile(scriptDir, 'pretrained_networks');
modelsDir = fullfile(scriptDir, 'models');
logsDir = fullfile(scriptDir, 'logs');
if ~exist(pretrainedDir, 'dir'), mkdir(pretrainedDir); end
if ~exist(modelsDir, 'dir'), mkdir(modelsDir); end
if ~exist(logsDir, 'dir'), mkdir(logsDir); end

% Set save paths with full paths
UAV_SAVE_PATH = fullfile(modelsDir, sprintf('uav_%s_%s.mat', RLAMPSO_MODE, timestamp));
LOG_FILE = fullfile(logsDir, sprintf('training_%s_%s.log', RLAMPSO_MODE, timestamp));

fprintf('✓ Directories created\n');
fprintf('✓ Log file: %s\n\n', LOG_FILE);

% Start diary logging to capture all output
diary(LOG_FILE);
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('TRAINING SESSION LOG\n');
fprintf('Started: %s\n', datestr(now));
fprintf('Log file: %s\n', LOG_FILE);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

%% =========================================================================
%  CONFIGURATION
%% =========================================================================

fprintf('Creating RLAMPSO configuration...\n');
config = RLAMPSO_Config(RLAMPSO_MODE);

% GPU detection and configuration
if canUseGPU()
    gpu = gpuDevice;
    config.useGPU = true;
    fprintf('✓ GPU detected: %s\n', gpu.Name);
    fprintf('  Available Memory: %.1f GB\n', gpu.AvailableMemory / 1e9);
    fprintf('  Batch Size (GPU-optimized): %d\n', config.batchSize);
else
    config.useGPU = false;
    fprintf('⚠ No GPU detected - using CPU (slower)\n');
    fprintf('  Batch Size (CPU): %d\n', config.batchSize);
end

fprintf('\n');
fprintf('════════ Training Configuration (from RLAMPSO_Config) ════════\n');
fprintf('Mode: %s\n', config.mode);
fprintf('\n  Features Enabled:\n');
fprintf('    Transformer: %s\n', bool2str(config.useTransformerState));
fprintf('    TD3: %s\n', bool2str(config.useTD3));
fprintf('    PER: %s\n', bool2str(config.usePER));
fprintf('    GPU: %s\n', bool2str(config.useGPU));
fprintf('    Parallel: %s (disabled for dlnetwork)\n', bool2str(config.useParallel));

fprintf('\n  Training Parameters:\n');
fprintf('    Episodes: %d\n', config.numEpisodes);
fprintf('    PSO Population: %d\n', config.popSize);
fprintf('    Max PSO Iterations: %d\n', config.maxIterations);
fprintf('    Batch Size: %d (GPU-optimized)\n', config.batchSize);
fprintf('    Actor LR: %.6f\n', config.actorLR);
fprintf('    Critic LR: %.6f\n', config.criticLR);

fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Estimate time (GPU sequential training)
if config.useGPU
    % GPU timing estimates (faster per episode)
    uavEstTime = config.numEpisodes * 120 / 3600;  % ~120 sec/episode on GPU
    fprintf('Estimated Time (GPU): ~%.1f hours (%d episodes)\n', uavEstTime, config.numEpisodes);
else
    % CPU timing estimates (slower per episode)
    uavEstTime = config.numEpisodes * 300 / 3600;  % ~300 sec/episode on CPU
    fprintf('Estimated Time (CPU): ~%.1f hours (%d episodes)\n', uavEstTime, config.numEpisodes);
end
fprintf('\n');

% Start training confirmation
try
    response = input('Start UAV training? (y/n): ', 's');
catch
    % Batch mode: auto-approve
    response = 'y';
    fprintf('(Batch mode: auto-approved)\n');
end
if ~strcmpi(response, 'y')
    fprintf('\nTraining cancelled.\n');
    diary off;
    return;
end
fprintf('\n');

%% =========================================================================
%  MAIN TRAINING (WITH ERROR HANDLING)
%% =========================================================================

try
    %% =========================================================================
    %  AGENT INITIALIZATION
    %% =========================================================================
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║            INITIALIZING AGENT FOR UAV TRAINING               ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    % Initialize new agent from scratch
    fprintf('Initializing new agent from scratch...\n');
    agent = initializeAgentAuto(config, '');
    fprintf('✓ Agent initialized\n\n');

    %% =========================================================================
    %  UAV PATH PLANNING TRAINING
    %% =========================================================================

    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║            UAV PATH PLANNING TRAINING                         ║\n');
    fprintf('║                  (%d episodes)                                   ║\n', config.numEpisodes);
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    fprintf('Training Configuration:\n');
    fprintf('  Episodes: %d\n', config.numEpisodes);
    fprintf('  Actor LR: %.6f\n', config.actorLR);
    fprintf('  Critic LR: %.6f\n\n', config.criticLR);

    % Update config.savePath to use the UAV save path
    config.savePath = UAV_SAVE_PATH;

    trainingStartTime = tic;

    % Run GPU-optimized sequential UAV training
    [trainedAgent, trainingStats] = trainRLAMPSO_Parallel(config, agent);

    trainingTime = toc(trainingStartTime);

    % Save trained agent
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                  SAVING TRAINED AGENT                         ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    uavMetadata = struct();
    uavMetadata.phase = 'UAV_Training';
    uavMetadata.episodes = config.numEpisodes;
    uavMetadata.config = config;
    uavMetadata.timestamp = datestr(now);
    uavMetadata.trainingTime = trainingTime;
    uavMetadata.trainingStats = trainingStats;

    save(UAV_SAVE_PATH, 'trainedAgent', 'uavMetadata', '-v7.3');

    fprintf('✓ Trained UAV agent saved to:\n  %s\n\n', UAV_SAVE_PATH);

    %% =========================================================================
    %  FINAL SUMMARY
    %% =========================================================================

    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║              UAV TRAINING COMPLETE!                           ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    fprintf('=== Training Summary ===\n');
    fprintf('Episodes: %d\n', config.numEpisodes);
    fprintf('Time: %.1f hours\n', trainingTime/3600);
    fprintf('Output: %s\n\n', UAV_SAVE_PATH);

    if exist('trainingStats', 'var')
        windowSize = min(20, config.numEpisodes);
        lastEpisodes = max(1, config.numEpisodes - windowSize + 1):config.numEpisodes;

        if length(trainingStats.episodeFitness) >= lastEpisodes(1)
            finalAvgFitness = mean(trainingStats.episodeFitness(lastEpisodes));
            initialFitness = trainingStats.episodeFitness(1);
            finalImprovement = initialFitness - finalAvgFitness;

            fprintf('=== Final UAV Performance (last %d episodes) ===\n', windowSize);
            fprintf('Average Fitness: %.2f\n', finalAvgFitness);
            fprintf('Cumulative Reward (Improvement): %.2f\n\n', finalImprovement);
        end
    end

    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('DEPLOYMENT\n');
    fprintf('═══════════════════════════════════════════════════════════════\n\n');

    fprintf('Load and use the trained agent:\n\n');
    fprintf('  loaded = load(''%s'');\n', UAV_SAVE_PATH);
    fprintf('  agent = loaded.trainedAgent;\n\n');
    fprintf('Deploy the agent:\n\n');
    fprintf('  [path, conv, params, stats] = globalPathPlanningRLAMPSO(...\n');
    fprintf('      startPoint, goalPoint, obstacles, trees, ...\n');
    fprintf('      terrainGrid, terrainX, terrainY, mapSize, ...\n');
    fprintf('      popSize, maxIterations, 0.9, 2.0, 2.0, ...\n');
    fprintf('      '''', config, agent);\n\n');

    fprintf('═══════════════════════════════════════════════════════════════\n\n');
    fprintf('✓ Training complete! Agent ready for deployment.\n\n');

catch ME
    % Training failed with error
    fprintf('\n');
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('TRAINING FAILED WITH ERROR\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  File: %s\n', ME.stack(i).file);
        fprintf('  Line: %d\n', ME.stack(i).line);
        fprintf('  Function: %s\n', ME.stack(i).name);
    end
    fprintf('═══════════════════════════════════════════════════════════════\n\n');

    % Stop logging
    diary off;

    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║              ERROR LOG SAVED                                  ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
    fprintf('Log file: %s\n\n', LOG_FILE);

    % Rethrow error
    rethrow(ME);
end

% Stop diary logging
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('TRAINING COMPLETED SUCCESSFULLY\n');
fprintf('Finished: %s\n', datestr(now));
fprintf('═══════════════════════════════════════════════════════════════\n\n');
diary off;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║              TRAINING LOG SAVED                               ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('Log file: %s\n\n', LOG_FILE);
fprintf('To view the log:\n');
fprintf('  type(''%s'')\n\n', LOG_FILE);

%% =========================================================================
%  HELPER FUNCTIONS
%% =========================================================================

function str = bool2str(value)
    if value
        str = 'ON';
    else
        str = 'OFF';
    end
end
