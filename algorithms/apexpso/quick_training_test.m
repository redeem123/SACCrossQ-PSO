% Quick Training Test for RRSACPSO
% Runs a short training session to observe behavior
%
% This will run 5 episodes with 100 iterations each

clc; clear;
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║         RRSACPSO Quick Training Test                    ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

% Add paths
scriptDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(scriptDir));

% Create minimal config for quick test
config = APEXPSO_Config();
config.numEpisodes = 5;           % Just 5 episodes
config.maxIterations = 100;       % 100 iterations per episode
config.warmupPeriod = 10;         % Short warmup
config.logInterval = 1;           % Log every episode
config.verbose = true;            % Show details

fprintf('Test Configuration:\n');
fprintf('  Episodes: %d\n', config.numEpisodes);
fprintf('  Iterations per episode: %d\n', config.maxIterations);
fprintf('  Population: %d particles\n', config.popSize);
fprintf('  Action space: %dD (%s)\n\n', config.actionSize, config.paramMode);

% Load terrain
try
    load('data/test_terrain.mat', 'terrainGrid', 'terrainX', 'terrainY');
    fprintf('✓ Loaded test terrain\n\n');
catch
    fprintf('⚠ Creating simple flat terrain\n\n');
    [terrainX, terrainY] = meshgrid(linspace(0, 400, 50), linspace(0, 400, 50));
    terrainGrid = zeros(size(terrainX));
end

% Simple test scenario
startPoint = [10, 10, 20];
goalPoint = [390, 390, 20];

% No danger zones for the smoke test.
dangerZones = [];

fprintf('Starting quick training test...\n');
fprintf('────────────────────────────────────────────────────────────\n\n');

tic;
try
    [bestPath, bestFitness, fitnessHistory, agent, stateEncoder] = ...
        globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, ...
        terrainGrid, terrainX, terrainY, config);

    elapsedTime = toc;

    % Display results
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                    Test Results                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');
    fprintf('  Final Best Fitness: %.4f\n', bestFitness);
    fprintf('  Training Time: %.2f seconds\n', elapsedTime);
    fprintf('  Time per Episode: %.2f seconds\n', elapsedTime / config.numEpisodes);
    fprintf('\n  Fitness History:\n');
    for i = 1:length(fitnessHistory)
        fprintf('    Episode %d: %.4f\n', i, fitnessHistory(i));
    end

    % Check for improvement
    improvement = fitnessHistory(1) - fitnessHistory(end);
    improvementPct = (improvement / fitnessHistory(1)) * 100;
    fprintf('\n  Improvement: %.4f (%.2f%%)\n', improvement, improvementPct);

    if improvement > 0
        fprintf('  Status: ✓ Agent is learning!\n');
    else
        fprintf('  Status: ⚠ No improvement yet (may need more episodes)\n');
    end

    % Plot fitness history
    figure('Name', 'RRSACPSO Training Progress');
    plot(1:length(fitnessHistory), fitnessHistory, 'b-o', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Best Fitness');
    title('RRSACPSO Learning Curve (Quick Test)');
    grid on;

    fprintf('\n✓ Test completed successfully!\n');
    fprintf('  If results look good, run full comparison: run_comparison\n\n');

catch ME
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                    Test Failed                           ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Location: %s (line %d)\n\n', ME.stack(1).name, ME.stack(1).line);

    % Display stack trace
    fprintf('  Stack trace:\n');
    for k = 1:min(5, length(ME.stack))
        fprintf('    %d. %s (line %d)\n', k, ME.stack(k).name, ME.stack(k).line);
    end
end
