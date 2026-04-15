function pretrain_sacpso_paper(numEpisodes, outputPath)
%PRETRAIN_SACPSO_PAPER Pre-train paper-aligned SAC-SAPSO on 45 benchmarks.
%
%   pretrain_sacpso_paper()           — 5000 episodes (from config)
%   pretrain_sacpso_paper(200)        — 200 episodes (override)
%   pretrain_sacpso_paper(200, path)  — custom output path
%
% Matches the SAC-SAPSO paper (von Eschwege & Engelbrecht 2024) algorithm:
%   - 3D global action (w, c1, c2 shared by all particles)
%   - ns+3 state (per-particle velocity + stability + infeasible + completion)
%   - Observation interval nt=25 (act every 25 iterations)
%   - Paper reward (Equation 25)
%   - Per-dimension velocity clamping (delta=0.5)
%
% Run with:
%   /Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -r "run('scripts/pretrain_sacpso_paper.m'); exit;"

    %% Setup paths
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(fullfile(repoRoot, 'data'));

    %% Parse arguments
    %% numEpisodes deferred until config is created
    if nargin < 2 || isempty(outputPath)
        outputPath = fullfile(repoRoot, 'models', 'AFSACPSO', 'pretrained_sacpso_paper.mat');
    end

    setenv('APEXPSO_CONFIG_QUIET', '1');

    %% PSO parameters — aligned with UAV deployment
    D = 30;           % All functions are 30-dimensional
    popSize = 100;    % Aligned with deployment popSize
    maxIter = 1000;   % Aligned with deployment maxIterations

    %% Load benchmark split
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
        error('pretrain_sacpso_paper:MissingFunctions', ...
            'The following functions failed validation: %s', strjoin(missingFuncs, ', '));
    end
    fprintf('  All %d training functions validated.\n\n', nTrain);

    %% Create SAC config (sacpso-paper mode)
    config = AFSACPSO_Config('sacpso-paper');
    config.popSize = popSize;
    config.maxIterations = maxIter;
    if nargin >= 1 && ~isempty(numEpisodes)
        config.numEpisodes = numEpisodes;
    end
    numEpisodes = config.numEpisodes;
    config.savePath = outputPath;
    config.saveInterval = 50;
    config.disableVisualization = true;
    config.mapSize = [1, 1, 1];  % dummy for config compatibility

    % Ensure stateSize matches popSize+3 for paper state
    config.stateSize = popSize + 3;

    %% Print summary
    fprintf('==============================================================\n');
    fprintf('  SAC-SAPSO Paper-Aligned Pre-Training\n');
    fprintf('==============================================================\n');
    fprintf('  Episodes:          %d (%.1f per function)\n', numEpisodes, numEpisodes/nTrain);
    fprintf('  Train functions:   %d\n', nTrain);
    fprintf('  Dimension:         %d\n', D);
    fprintf('  Swarm size:        %d\n', popSize);
    fprintf('  Iterations/ep:     %d\n', maxIter);
    fprintf('  Action:            %dD (global w,c1,c2)\n', config.actionSize);
    fprintf('  State:             %dD (ns+3 paper state)\n', config.stateSize);
    fprintf('  Observation int:   %d\n', config.observationInterval);
    fprintf('  Reward:            paper (Eq. 25)\n');
    fprintf('  Vel clamp:         per-dimension (delta=%.2f)\n', config.velocityClampDelta);
    fprintf('  Networks:          [%s] / [%s]\n', ...
        num2str(config.actorHiddenLayers), num2str(config.criticHiddenLayers));
    fprintf('  Gamma:             %.2f\n', config.gamma);
    fprintf('  Tau:               %.4f\n', config.tau);
    fprintf('  LR:                actor=%.1e, critic=%.1e\n', config.actorLR, config.criticLR);
    fprintf('  UTD:               %d\n', config.utdRatio);
    fprintf('  Batch:             %d\n', config.batchSize);
    fprintf('  Buffer:            %d\n', config.bufferSize);
    fprintf('  Save path:         %s\n', outputPath);
    fprintf('==============================================================\n\n');

    %% Initialize agent (state encoder not used in paper mode)
    rng(42, 'twister');
    agent = AFSACPSO_Agent(config);
    stateEncoder = [];  % Not used in paper state mode

    %% Training loop
    trainingTimer = tic;
    cpuTimerStart = cputime;
    fitnessHistory = zeros(1, numEpisodes);
    funcAssignment = zeros(1, numEpisodes);

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

        epTimer = tic;
        [~, bestFit, ~] = psoMinimizeBenchmark( ...
            info.f, D, info.lb, info.ub, agent, stateEncoder, config);
        epTime = toc(epTimer);

        fitnessHistory(episode) = bestFit;

        if mod(episode, 10) == 0
            fprintf('[Ep %4d/%d] %-28s  best=%.4g  alpha=%.4f  buffer=%d  (%.1fs)\n', ...
                episode, numEpisodes, funcName, bestFit, ...
                exp(agent.logAlpha), agent.replayBuffer.size, epTime);
        end

        % Periodic checkpoint
        if mod(episode, config.saveInterval) == 0
            agent.saveWeights(outputPath);
            fprintf('  [Checkpoint] Saved weights at episode %d\n', episode);
        end
    end

    totalTime = toc(trainingTimer);
    totalCpuTime = cputime - cpuTimerStart;

    %% Save final weights
    agent.saveWeights(outputPath);
    fprintf('\n  [Final] Saved trained weights to %s\n', outputPath);

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
        title(sprintf('SAC-SAPSO Paper Pre-Training (%d episodes, %d functions)', ...
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

        plotPath = fullfile(saveDir, 'pretrain_sacpso_paper_convergence.pdf');
        exportgraphics(fig, plotPath, 'ContentType', 'vector');
        close(fig);
        fprintf('  Convergence plot saved to %s\n', plotPath);
    catch e
        fprintf('  Warning: Could not save plot: %s\n', e.message);
    end

    %% Summary
    fprintf('\n');
    fprintf('==============================================================\n');
    fprintf('  PRE-TRAINING COMPLETE (SAC-SAPSO Paper-Aligned)\n');
    fprintf('==============================================================\n');
    fprintf('  Total time:      %.1f minutes (%.1f hours)\n', totalTime/60, totalTime/3600);
    fprintf('  Episodes:        %d\n', numEpisodes);
    fprintf('  Weights:         %s\n', outputPath);
    fprintf('\nNext: run UAV ablation:\n');
    fprintf('  VIETANH_ALGO_GROUP=sacpso-paper VIETANH_NUM_RUNS=30 \\\n');
    fprintf('    VIETANH_SCENARIO_FILTER="Simple" APEXPSO_CONFIG_QUIET=1 \\\n');
    fprintf('    /Applications/MATLAB_R2024a.app/bin/matlab -nodisplay -r "run(''run_comparison.m''); exit;"\n');
end
