function pretrain_rlampso(numEpisodes, outputPath)
%PRETRAIN_RLAMPSO Pre-train RLAMPSO DDPG on 45 benchmark functions.
%
%   pretrain_rlampso()           — 5000 episodes (from config)
%   pretrain_rlampso(450)        — 450 episodes (override)
%   pretrain_rlampso(200, path)  — custom output path
%
% Uses the same 45-function training set (getBenchmarkSplit) as the
% AFSACPSO SAC pre-training, so both algorithms get a fair comparison.
%
% The DDPG agent (Yin et al. 2023) learns to adapt 5-subgroup PSO
% parameters [w, c1, c2] via a 20D action (5 subgroups x 4 dims)
% conditioned on a 15D sin-encoded state.
%
% Run with:
%   /Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -r "run('scripts/pretrain_rlampso.m'); exit;"

    %% Setup paths
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(fullfile(repoRoot, 'data'));

    %% Parse arguments (numEpisodes deferred until config is created)
    if nargin < 2 || isempty(outputPath)
        outputDir = fullfile(repoRoot, 'models', 'rlampso');
        if ~exist(outputDir, 'dir'), mkdir(outputDir); end
        outputPath = fullfile(outputDir, 'rlampso.mat');
    end

    %% PSO parameters — aligned with deployment
    D = 30;

    %% Load benchmark split (same 45 functions as AFSACPSO)
    split = getBenchmarkSplit();
    trainNames = split.train;
    nTrain = numel(trainNames);

    %% Validate all training functions
    fprintf('\n');
    fprintf('==============================================================\n');
    fprintf('  Validating %d training functions (D=%d)...\n', nTrain, D);
    fprintf('==============================================================\n');
    missingFuncs = {};
    for i = 1:nTrain
        try
            info = getBenchmarkFunction(trainNames{i}, D);
            testX = (info.lb + info.ub) / 2;
            fval = info.f(testX);
            if ~isfinite(fval)
                warning('Function %s returned non-finite at midpoint.', trainNames{i});
            end
            fprintf('  [OK] %2d. %-30s bounds=[%.1f, %.1f]  f(mid)=%.4g\n', ...
                i, trainNames{i}, info.lb(1), info.ub(1), fval);
        catch e
            fprintf('  [FAIL] %2d. %-30s ERROR: %s\n', i, trainNames{i}, e.message);
            missingFuncs{end+1} = trainNames{i}; %#ok<AGROW>
        end
    end

    if ~isempty(missingFuncs)
        error('pretrain_rlampso:MissingFunctions', ...
            'The following functions failed validation: %s', strjoin(missingFuncs, ', '));
    end
    fprintf('  All %d training functions validated.\n\n', nTrain);

    %% Create DDPG config (pretrain mode: deployment-aligned PSO budget)
    config = RLAMPSO_Config('pretrain');
    if nargin >= 1 && ~isempty(numEpisodes)
        config.numEpisodes = numEpisodes;
    end
    numEpisodes = config.numEpisodes;
    popSize = config.popSize;
    maxIter = config.maxIterations;
    config.verbose = false;

    %% Print summary
    fprintf('==============================================================\n');
    fprintf('  RLAMPSO DDPG Pre-Training on Benchmark Functions\n');
    fprintf('==============================================================\n');
    fprintf('  Episodes:          %d (%.1f per function)\n', numEpisodes, numEpisodes/nTrain);
    fprintf('  Train functions:   %d\n', nTrain);
    fprintf('  Dimension:         %d\n', D);
    fprintf('  Swarm size:        %d\n', popSize);
    fprintf('  Iterations/ep:     %d\n', maxIter);
    fprintf('  State size:        %d\n', config.stateSize);
    fprintf('  Action size:       %d (5 subgroups x 4)\n', config.actionSize);
    fprintf('  Actor layers:      [%s]\n', num2str(config.actorHiddenLayers));
    fprintf('  Critic layers:     [%s]\n', num2str(config.criticHiddenLayers));
    fprintf('  Gamma:             %.2f\n', config.gamma);
    fprintf('  Tau:               %.4f\n', config.tau);
    fprintf('  LR:                actor=%.1e, critic=%.1e\n', config.actorLR, config.criticLR);
    fprintf('  Noise:             N(0, %.1f)\n', config.explorationNoise);
    fprintf('  Batch size:        %d\n', config.batchSize);
    fprintf('  Buffer size:       %d\n', config.bufferSize);
    fprintf('  Save path:         %s\n', outputPath);
    fprintf('==============================================================\n\n');

    %% Initialize DDPG agent
    rng(42, 'twister');
    agent = initializeAgentAuto(config, '');

    %% Training loop — round-robin over 45 functions
    trainingTimer = tic;
    cpuTimerStart = cputime;
    fitnessHistory = zeros(1, numEpisodes);
    funcAssignment = zeros(1, numEpisodes);
    saveInterval = 50;

    for episode = 1:numEpisodes
        % Cycle through training functions (round-robin with shuffle)
        funcIdx = mod(episode-1, nTrain) + 1;
        if funcIdx == 1 && episode > 1
            perm = randperm(nTrain);
            trainNames = split.train(perm);
        end
        funcAssignment(episode) = funcIdx;

        funcName = trainNames{funcIdx};
        info = getBenchmarkFunction(funcName, D);

        % Run one PSO episode with online DDPG training
        % Wrap handle: runRLAMPSOonBenchmark passes column vectors,
        % but getBenchmarkFunction expects row vectors.
        fitFcn = @(x) info.f(x');
        epTimer = tic;
        [bestFit, ~, agent] = runRLAMPSOonBenchmark( ...
            fitFcn, D, info.lb(1), info.ub(1), info.fmin, config, agent);
        epTime = toc(epTimer);

        fitnessHistory(episode) = bestFit;

        if mod(episode, 10) == 0
            fprintf('[Ep %4d/%d] %-28s  best=%.4g  buffer=%d  (%.1fs)\n', ...
                episode, numEpisodes, funcName, bestFit, ...
                length(agent.replayBuffer), epTime);
        end

        % Periodic checkpoint
        if mod(episode, saveInterval) == 0
            trainedAgent = agent; %#ok<NASGU>
            save(outputPath, 'trainedAgent', '-v7.3');
            fprintf('  [Checkpoint] Saved at episode %d\n', episode);
        end
    end

    totalTime = toc(trainingTimer);
    totalCpuTime = cputime - cpuTimerStart;

    %% Save final weights
    trainedAgent = agent; %#ok<NASGU>
    save(outputPath, 'trainedAgent', '-v7.3');
    fprintf('\n  [Final] Saved trained DDPG agent to %s\n', outputPath);

    %% Save training log
    [saveDir, baseName, ~] = fileparts(outputPath);
    logPath = fullfile(saveDir, [baseName '_log.mat']);
    save(logPath, 'fitnessHistory', 'funcAssignment', 'trainNames', ...
         'numEpisodes', 'D', 'popSize', 'maxIter', 'totalTime', 'totalCpuTime', '-v7.3');
    fprintf('  Training log saved to %s\n', logPath);

    %% Save convergence plot
    try
        fig = figure('Visible', 'off');
        subplot(2,1,1);
        plot(1:numEpisodes, fitnessHistory, '.', 'MarkerSize', 2);
        xlabel('Episode'); ylabel('Best Fitness');
        title(sprintf('RLAMPSO DDPG Benchmark Pre-Training (%d episodes, %d functions)', ...
            numEpisodes, nTrain));
        grid on;

        subplot(2,1,2);
        windowSize = nTrain;
        if numEpisodes >= windowSize
            runMean = movmean(fitnessHistory, windowSize);
            plot(1:numEpisodes, runMean, 'r-', 'LineWidth', 1.5);
            xlabel('Episode'); ylabel('Running Mean Fitness');
            title(sprintf('Smoothed (window=%d)', windowSize));
            grid on;
        end

        plotPath = fullfile(saveDir, 'pretrain_rlampso_convergence.pdf');
        exportgraphics(fig, plotPath, 'ContentType', 'vector');
        close(fig);
        fprintf('  Convergence plot saved to %s\n', plotPath);
    catch e
        fprintf('  Warning: Could not save plot: %s\n', e.message);
    end

    %% Summary
    fprintf('\n');
    fprintf('==============================================================\n');
    fprintf('  RLAMPSO PRE-TRAINING COMPLETE\n');
    fprintf('==============================================================\n');
    fprintf('  Total time:      %.1f minutes (%.1f hours)\n', totalTime/60, totalTime/3600);
    fprintf('  Episodes:        %d\n', numEpisodes);
    fprintf('  Weights:         %s\n', outputPath);
    fprintf('\nNext: run benchmark comparison with pretrained agent:\n');
    fprintf('  VIETANH_ALGO_GROUP=rlbased VIETANH_NUM_RUNS=10 \\\n');
    fprintf('    /Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -r "run_comparison; exit"\n');
end
