%% CLEAN_AND_RETRAIN.m
% Script to delete contaminated pretrained agents and retrain from scratch
%
% WHY THIS IS NEEDED:
%   The pretrained agent was trained with buggy reward function that gave
%   negative rewards for normal PSO stagnation. The replay buffer and
%   network weights are contaminated with this bad training data.
%
%   We MUST start fresh to see if the sparse reward fix works.
%
% WHAT THIS DOES:
%   1. Clears MATLAB cache and workspace
%   2. Deletes all saved agent files
%   3. Runs train_advanced with fresh initialization

%% Step 1: Clear everything
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     CLEANING CONTAMINATED AGENTS AND RETRAINING FROM SCRATCH  ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('Step 1: Clearing MATLAB workspace and cache...\n');
clear all;
clc;
close all;
rehash toolboxcache;
fprintf('✓ Workspace cleared\n\n');

%% Step 2: Delete contaminated agent files
fprintf('Step 2: Deleting contaminated pretrained agent files...\n');

% Check if saved_agents directory exists
savedAgentsDir = 'saved_agents';
if exist(savedAgentsDir, 'dir')
    % Find all .mat files in saved_agents
    agentFiles = dir(fullfile(savedAgentsDir, '*.mat'));

    if ~isempty(agentFiles)
        fprintf('  Found %d agent file(s) to delete:\n', length(agentFiles));
        for i = 1:length(agentFiles)
            filepath = fullfile(savedAgentsDir, agentFiles(i).name);
            fprintf('    - %s\n', agentFiles(i).name);
            delete(filepath);
        end
        fprintf('✓ All contaminated agent files deleted\n\n');
    else
        fprintf('  No agent files found (directory is empty)\n\n');
    end
else
    fprintf('  saved_agents directory does not exist (nothing to delete)\n\n');
end

%% Step 3: Confirm retraining
fprintf('Step 3: Ready to retrain from scratch with fixed reward function\n\n');

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('CHANGES IN FIXED REWARD FUNCTION:\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  OLD (WRONG):  Stagnation → reward = -0.1  (83%% negative!)\n');
fprintf('  NEW (FIXED):  Stagnation → reward =  0.0  (sparse rewards)\n\n');
fprintf('  This prevents policy collapse from negative reward bias.\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

response = input('Start fresh training with fixed rewards? (y/n): ', 's');
if ~strcmpi(response, 'y')
    fprintf('\nCancelled. No training started.\n');
    return;
end

%% Step 4: Run train_advanced
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║          STARTING FRESH TRAINING (NO PRETRAINED AGENT)        ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

% Run the training script
train_advanced;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    TRAINING COMPLETE                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('Check the training log to verify:\n');
fprintf('  ✓ Actions NOT saturating (should vary, not constant -1/+1)\n');
fprintf('  ✓ PSO parameters adapting (c1 and c2 not collapsing to 0)\n');
fprintf('  ✓ Fitness decreasing over episodes (not increasing)\n');
fprintf('  ✓ Critic loss staying low (< 0.01, not exploding)\n\n');
