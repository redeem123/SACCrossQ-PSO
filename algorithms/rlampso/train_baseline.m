% =========================================================================
% RLAMPSO BASELINE TRAINING (Paper-exact DDPG)
% =========================================================================
% Pre-trains the DDPG agent on CEC 2013 benchmarks or UAV scenarios.
%
% Uses RLAMPSO_Config('baseline') for all hyperparameters.
% Output: trained agent .mat file for deployment.
% =========================================================================

clear; clc; close all;

fprintf('\nRLAMPSO Baseline Training\n\n');

%% Setup
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
setupPaths();

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
modelsDir = fullfile(scriptDir, 'models');
logsDir   = fullfile(scriptDir, 'logs');
if ~exist(modelsDir, 'dir'), mkdir(modelsDir); end
if ~exist(logsDir,   'dir'), mkdir(logsDir);   end

SAVE_PATH = fullfile(modelsDir, sprintf('rlampso_baseline_%s.mat', timestamp));
LOG_FILE  = fullfile(logsDir,   sprintf('training_baseline_%s.log', timestamp));

diary(LOG_FILE);
fprintf('Log: %s\n', LOG_FILE);

%% Configuration
config = RLAMPSO_Config('baseline');
config.verbose = true;

fprintf('Mode: %s\n', config.mode);
fprintf('Episodes: %d | PopSize: %d | MaxIter: %d\n', ...
    config.numEpisodes, config.popSize, config.maxIterations);
fprintf('Actor LR: %.5f | Critic LR: %.5f | Tau: %.4f\n', ...
    config.actorLR, config.criticLR, config.tau);

%% Initialize agent
fprintf('\nInitializing DDPG agent...\n');
agent = initializeAgentAuto(config, '');
fprintf('Agent ready.\n\n');

%% Train on CEC benchmarks
fprintf('Training on CEC 2013 benchmarks...\n\n');
trainingStart = tic;

functionIDs = 1:28;
allStats = struct();

for fIdx = 1:length(functionIDs)
    fID = functionIDs(fIdx);
    fprintf('\n=== Function F%d (%d/%d) ===\n', fID, fIdx, length(functionIDs));

    [agent, stats] = trainCECFunction_Parallel(fID, config.numEpisodes, config, agent);
    allStats(fIdx).functionID = fID;
    allStats(fIdx).stats = stats;
end

trainingTime = toc(trainingStart);

%% Save
trainedAgent = agent;
metadata = struct('timestamp', datestr(now), 'trainingTime', trainingTime, ...
                  'config', config, 'allStats', allStats);
save(SAVE_PATH, 'trainedAgent', 'metadata', '-v7.3');
fprintf('\nSaved: %s\n', SAVE_PATH);
fprintf('Total time: %.1f hours\n', trainingTime/3600);

diary off;
