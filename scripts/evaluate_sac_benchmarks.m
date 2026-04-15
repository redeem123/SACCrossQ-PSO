function evaluate_sac_benchmarks(weightsPath, numRuns, outputDir)
%EVALUATE_SAC_BENCHMARKS Evaluate pre-trained SAC on train+test benchmark functions.
%
%   evaluate_sac_benchmarks()
%   evaluate_sac_benchmarks('models/AFSACPSO/pretrained_sac_benchmarks.mat', 30)
%
% Evaluates the frozen learned policy on both:
%   - 45 training functions (in-distribution)
%   - 7 held-out test functions (generalisation)
% Also runs a no-SAC baseline (constant w=0.5, c1=1.5, c2=1.5) for comparison.
%
% Paper protocol: D=30, swarm=30, maxIter=5000, frozen policy (no learning).

    %% Setup paths
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(fullfile(repoRoot, 'data'));

    %% Parse arguments
    if nargin < 1 || isempty(weightsPath)
        weightsPath = fullfile(repoRoot, 'models', 'AFSACPSO', 'pretrained_sac_benchmarks.mat');
    end
    if nargin < 2 || isempty(numRuns)
        numRuns = 30;
    end
    if nargin < 3 || isempty(outputDir)
        timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        outputDir = fullfile(repoRoot, 'outputs', 'research', ...
            ['sac_benchmark_eval_' timestamp]);
    end

    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end

    setenv('APEXPSO_CONFIG_QUIET', '1');

    %% PSO parameters — aligned with training and UAV deployment
    D = 30;
    popSize = 100;
    maxIter = 1000;

    %% Validate weights exist
    if ~exist(weightsPath, 'file')
        error('Weights file not found: %s\nRun pretrain_sac_benchmarks.m first.', weightsPath);
    end

    %% Load benchmark split
    split = getBenchmarkSplit();
    allFuncs = [split.train, split.test];
    nTrain = numel(split.train);
    nTest = numel(split.test);
    nTotal = nTrain + nTest;

    %% Validate all functions
    fprintf('\n');
    fprintf('==============================================================\n');
    fprintf('  Validating %d functions (D=%d)...\n', nTotal, D);
    fprintf('==============================================================\n');
    for i = 1:nTotal
        info = getBenchmarkFunction(allFuncs{i}, D);
        testX = (info.lb + info.ub) / 2;
        fval = info.f(testX);
        setLabel = 'TRAIN';
        if i > nTrain, setLabel = 'TEST '; end
        fprintf('  [%s] %2d. %-30s f(mid)=%.4g\n', setLabel, i, allFuncs{i}, fval);
    end
    fprintf('\n');

    %% Build config for frozen evaluation
    config = AFSACPSO_Config('pretrain');
    config.popSize = popSize;
    config.maxIterations = maxIter;
    config.numEpisodes = 1;
    config.paramMode = 'attractor-field';
    config.actionSize = 9;
    config.stateSize = 15;
    config.temporalWindow = 8;
    config.paramsPerParticle = 3;
    config.mapSize = [1,1,1];
    config.disableVisualization = true;
    config.usePerParticleActions = false;
    config.useRankResidualControl = true;
    config.targetEntropy = -config.actionSize;
    config.warmupPeriod = maxIter + 1;  % Freeze: no gradient updates
    config.trainEveryNIterations = 1;
    config.actorHiddenLayers = [256, 256];  % Paper: actor 256
    config.criticHiddenLayers = [256, 256]; % Paper: critic 256
    config.gamma = 1.0;                    % Paper: gamma = 1
    config.tau = 0.005;                    % Paper: tau = 0.005

    %% Print summary
    fprintf('==============================================================\n');
    fprintf('  SAC Benchmark Evaluation\n');
    fprintf('==============================================================\n');
    fprintf('  Weights:         %s\n', weightsPath);
    fprintf('  Train funcs:     %d\n', nTrain);
    fprintf('  Test funcs:      %d\n', nTest);
    fprintf('  Runs per func:   %d\n', numRuns);
    fprintf('  D=%d, pop=%d, maxIter=%d\n', D, popSize, maxIter);
    fprintf('==============================================================\n\n');

    %% Evaluate both SAC and no-SAC baseline
    methods = {'SAC', 'NoSAC'};
    results = struct();

    for m = 1:numel(methods)
        methodName = methods{m};
        fprintf('\n--- Evaluating method: %s ---\n', methodName);

        funcResults = zeros(nTotal, numRuns);

        for fi = 1:nTotal
            funcName = allFuncs{fi};
            info = getBenchmarkFunction(funcName, D);

            for run = 1:numRuns
                rng(run * 1000 + fi * 100, 'twister');

                % Create fresh agent per run
                agent = AFSACPSO_Agent(config);
                stateEncoder = SimpleStateEncoder(config);

                if strcmp(methodName, 'SAC')
                    agent.loadWeights(weightsPath);
                    runConfig = config;
                    runConfig.forcedZeroAction = false;
                else
                    runConfig = config;
                    runConfig.forcedZeroAction = true;
                end

                [~, bestFit, ~] = psoMinimizeBenchmark( ...
                    info.f, D, info.lb, info.ub, agent, stateEncoder, runConfig);

                funcResults(fi, run) = bestFit;
            end

            setLabel = 'TRAIN';
            if fi > nTrain, setLabel = 'TEST '; end
            fprintf('  [%s] %-28s  mean=%.4g +/- %.4g\n', setLabel, funcName, ...
                mean(funcResults(fi,:)), std(funcResults(fi,:)));
        end

        results.(methodName) = funcResults;
    end

    %% Compute summary statistics
    trainResults = struct();
    testResults = struct();

    for m = 1:numel(methods)
        mName = methods{m};
        trainResults.(mName) = results.(mName)(1:nTrain, :);
        testResults.(mName) = results.(mName)(nTrain+1:end, :);
    end

    %% Write CSV: per-function summary
    writeResultsCSV(fullfile(outputDir, 'train_results.csv'), ...
        split.train, trainResults, methods, numRuns);
    writeResultsCSV(fullfile(outputDir, 'test_results.csv'), ...
        split.test, testResults, methods, numRuns);
    writeResultsCSV(fullfile(outputDir, 'all_results.csv'), ...
        allFuncs, results, methods, numRuns);

    %% Write summary table
    writeSummaryTable(fullfile(outputDir, 'summary.csv'), ...
        split, results, methods, nTrain, numRuns);

    %% Save raw data
    save(fullfile(outputDir, 'evaluation_results.mat'), ...
        'results', 'split', 'methods', 'D', 'popSize', 'maxIter', ...
        'numRuns', 'weightsPath', '-v7.3');

    %% Print final summary
    fprintf('\n');
    fprintf('==============================================================\n');
    fprintf('  EVALUATION COMPLETE\n');
    fprintf('==============================================================\n');
    for m = 1:numel(methods)
        mName = methods{m};
        trainMeans = mean(results.(mName)(1:nTrain, :), 2);
        testMeans = mean(results.(mName)(nTrain+1:end, :), 2);
        fprintf('  %s:\n', mName);
        fprintf('    Train (45): mean of means = %.4g\n', mean(trainMeans));
        fprintf('    Test  (7):  mean of means = %.4g\n', mean(testMeans));
    end

    % Pairwise comparison: how often does SAC beat NoSAC?
    sacBetterTrain = 0; sacBetterTest = 0;
    for fi = 1:nTrain
        sacMean = mean(results.SAC(fi,:));
        noSacMean = mean(results.NoSAC(fi,:));
        if sacMean < noSacMean, sacBetterTrain = sacBetterTrain + 1; end
    end
    for fi = 1:nTest
        sacMean = mean(results.SAC(nTrain+fi,:));
        noSacMean = mean(results.NoSAC(nTrain+fi,:));
        if sacMean < noSacMean, sacBetterTest = sacBetterTest + 1; end
    end
    fprintf('\n  SAC wins (lower mean):\n');
    fprintf('    Train: %d/%d functions\n', sacBetterTrain, nTrain);
    fprintf('    Test:  %d/%d functions\n', sacBetterTest, nTest);
    fprintf('\n  Results saved to: %s\n', outputDir);
end

%% --- Helper functions ---

function writeResultsCSV(filePath, funcNames, results, methods, numRuns)
    fid = fopen(filePath, 'w');
    cleanup = onCleanup(@() fclose(fid));

    % Header
    cols = {'Function'};
    for m = 1:numel(methods)
        cols{end+1} = [methods{m} '_mean']; %#ok<AGROW>
        cols{end+1} = [methods{m} '_std'];  %#ok<AGROW>
        cols{end+1} = [methods{m} '_median']; %#ok<AGROW>
    end
    fprintf(fid, '%s\n', strjoin(cols, ','));

    % Data
    for fi = 1:numel(funcNames)
        fprintf(fid, '%s', funcNames{fi});
        for m = 1:numel(methods)
            vals = results.(methods{m})(fi, :);
            fprintf(fid, ',%.10g,%.10g,%.10g', mean(vals), std(vals), median(vals));
        end
        fprintf(fid, '\n');
    end
end

function writeSummaryTable(filePath, split, results, methods, nTrain, numRuns)
    fid = fopen(filePath, 'w');
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'Set,Metric');
    for m = 1:numel(methods)
        fprintf(fid, ',%s', methods{m});
    end
    fprintf(fid, '\n');

    sets = {'Train', 'Test'};
    ranges = {1:nTrain, nTrain+1:nTrain+numel(split.test)};

    for s = 1:2
        setName = sets{s};
        idxRange = ranges{s};
        for m = 1:numel(methods)
            allMeans(m) = mean(mean(results.(methods{m})(idxRange, :), 2)); %#ok<AGROW>
            allStds(m) = mean(std(results.(methods{m})(idxRange, :), 0, 2)); %#ok<AGROW>
        end
        fprintf(fid, '%s,Mean of means', setName);
        for m = 1:numel(methods)
            fprintf(fid, ',%.10g', allMeans(m));
        end
        fprintf(fid, '\n');

        fprintf(fid, '%s,Mean of stds', setName);
        for m = 1:numel(methods)
            fprintf(fid, ',%.10g', allStds(m));
        end
        fprintf(fid, '\n');
    end
end
