function evaluate_pretrained_heldout_comparison(numRuns, outputDir)
%EVALUATE_PRETRAINED_HELDOUT_COMPARISON Compare pretrained controllers on held-out test functions.
%
%   evaluate_pretrained_heldout_comparison()
%   evaluate_pretrained_heldout_comparison(30)
%   evaluate_pretrained_heldout_comparison(30, 'outputs/research/my_eval')
%
% Runs the 7 held-out benchmark functions from getBenchmarkSplit() using:
%   - 100 particles
%   - 1000 PSO iterations
%   - numRuns repeated trials per function
%
% Evaluated variants:
%   - Attractor-field AFSACPSO
%   - Global SAC controller
%   - 5-subgroup SAC controller
%   - Per-particle SAC controller
%   - SAC-SAPSO (paper mode)
%   - RLAMPSO
%
% This script checkpoints after every function and can resume if re-run.

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(fullfile(repoRoot, 'data'));
    setenv('APEXPSO_CONFIG_QUIET', '1');

    if nargin < 1 || isempty(numRuns)
        numRuns = 30;
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = fullfile(repoRoot, 'outputs', 'research', 'heldout_eval_all_pretrained_30runs');
    end
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end

    progressPath = fullfile(outputDir, 'heldout_eval_progress.mat');
    resultsPath = fullfile(outputDir, 'heldout_eval_all_pretrained_results.mat');
    summaryPath = fullfile(outputDir, 'heldout_eval_summary.csv');

    D = 30;
    popSize = 100;
    maxIter = 1000;
    split = getBenchmarkSplit();
    testFuncs = split.test;
    nFuncs = numel(testFuncs);

    variants = { ...
        struct('name','RLAMPSO', ...
               'kind','rlampso', ...
               'weights',fullfile(repoRoot,'models','rlampso','pretrained_rlampso.mat')), ...
        struct('name','Attractor-field AFSACPSO', ...
               'kind','sac', ...
               'weights',fullfile(repoRoot,'models','AFSACPSO','pretrained_sac_attractor.mat'), ...
               'paramMode','attractor-field', ...
               'stateMode','rich', ...
               'rewardMode','rich'), ...
        struct('name','Global SAC controller', ...
               'kind','sac', ...
               'weights',fullfile(repoRoot,'models','AFSACPSO','pretrained_sac_global.mat'), ...
               'paramMode','global', ...
               'stateMode','rich', ...
               'rewardMode','rich'), ...
        struct('name','5-subgroup SAC controller', ...
               'kind','sac', ...
               'weights',fullfile(repoRoot,'models','AFSACPSO','pretrained_sac_benchmarks_5subgroup.mat'), ...
               'paramMode','5subgroup', ...
               'stateMode','rich', ...
               'rewardMode','rich'), ...
        struct('name','Per-particle SAC controller', ...
               'kind','sac', ...
               'weights',fullfile(repoRoot,'models','AFSACPSO','pretrained_sac_perparticle.mat'), ...
               'paramMode','per-particle', ...
               'stateMode','rich', ...
               'rewardMode','rich'), ...
        struct('name','SAC-SAPSO', ...
               'kind','sac', ...
               'weights',fullfile(repoRoot,'models','AFSACPSO','pretrained_sacpso_paper.mat'), ...
               'paramMode','global', ...
               'stateMode','paper', ...
               'rewardMode','paper') ...
    };

    if exist(progressPath, 'file') == 2
        loaded = load(progressPath);
        results = loaded.results;
        cpuTimes = loaded.cpuTimes;
        completed = loaded.completed;
        if isfield(loaded, 'statuses')
            statuses = loaded.statuses;
        else
            statuses = struct();
        end
        fprintf('Resuming held-out evaluation from %s\n', progressPath);
    else
        results = struct();
        cpuTimes = struct();
        completed = struct();
        statuses = struct();
        for vi = 1:numel(variants)
            field = matlab.lang.makeValidName(variants{vi}.name);
            results.(field) = nan(nFuncs, numRuns);
            cpuTimes.(field) = nan(nFuncs, 1);
            completed.(field) = false(nFuncs, 1);
            statuses.(field) = "pending";
        end
    end

    for vi = 1:numel(variants)
        field = matlab.lang.makeValidName(variants{vi}.name);
        if ~isfield(statuses, field)
            statuses.(field) = "pending";
        end
    end

    for vi = 1:numel(variants)
        v = variants{vi};
        field = matlab.lang.makeValidName(v.name);
        if strcmp(statuses.(field), "failed")
            fprintf('\n=== Retrying %s after previous failure ===\n', v.name);
            statuses.(field) = "pending";
        end
        fprintf('\n=== Evaluating %s ===\n', v.name);

        try
            for fi = 1:nFuncs
                if completed.(field)(fi)
                    fprintf('  [%d/%d] %s already completed; skipping.\n', fi, nFuncs, testFuncs{fi});
                    continue;
                end

                funcName = testFuncs{fi};
                info = getBenchmarkFunction(funcName, D);
                fprintf('  [%d/%d] %s\n', fi, nFuncs, funcName);

                cpuStart = cputime;
                runVals = nan(1, numRuns);
                for run = 1:numRuns
                    rng(run * 1000 + fi * 100, 'twister');
                    runVals(run) = evaluateOneRun(v, info, D, popSize, maxIter);
                end
                cpuElapsed = cputime - cpuStart;

                results.(field)(fi, :) = runVals;
                cpuTimes.(field)(fi) = cpuElapsed;
                completed.(field)(fi) = true;

                fprintf('    mean=%.6g std=%.6g cpu=%.3fs\n', mean(runVals), std(runVals), cpuElapsed);
                statuses.(field) = "completed";
                save(progressPath, 'results', 'cpuTimes', 'completed', 'statuses', 'variants', 'testFuncs', 'numRuns', 'D', 'popSize', 'maxIter', '-v7.3');
            end
        catch err
            statuses.(field) = "failed";
            warning('Variant %s failed: %s', v.name, err.message);
            save(progressPath, 'results', 'cpuTimes', 'completed', 'statuses', 'variants', 'testFuncs', 'numRuns', 'D', 'popSize', 'maxIter', '-v7.3');
        end
    end

    summary = struct('name',{},'mean_all',{},'std_all',{},'cpu_time_sec',{},'status',{});
    for vi = 1:numel(variants)
        field = matlab.lang.makeValidName(variants{vi}.name);
        vals = results.(field)(:);
        vals = vals(isfinite(vals));
        summary(vi).name = variants{vi}.name;
        if isempty(vals)
            summary(vi).mean_all = NaN;
            summary(vi).std_all = NaN;
        else
            summary(vi).mean_all = mean(vals);
            summary(vi).std_all = std(vals);
        end
        summary(vi).cpu_time_sec = sum(cpuTimes.(field), 'omitnan');
        summary(vi).status = char(statuses.(field));
    end

    save(resultsPath, 'results', 'summary', 'cpuTimes', 'statuses', 'variants', 'testFuncs', 'numRuns', 'D', 'popSize', 'maxIter', '-v7.3');
    writeSummaryCsv(summaryPath, summary);

    fprintf('\nSaved held-out results to:\n  %s\n  %s\n', resultsPath, summaryPath);
end

function bestFit = evaluateOneRun(variant, info, D, popSize, maxIter)
    if strcmp(variant.kind, 'sac')
        config = AFSACPSO_Config('pretrain');
        config.popSize = popSize;
        config.maxIterations = maxIter;
        config.numEpisodes = 1;
        config.paramsPerParticle = 3;
        config.mapSize = [1, 1, 1];
        config.disableVisualization = true;
        config.warmupPeriod = maxIter + 1;
        config.trainEveryNIterations = 1;
        config.actorHiddenLayers = [256, 256];
        config.criticHiddenLayers = [256, 256];
        config.gamma = 1.0;
        config.tau = 0.005;
        config.paramMode = variant.paramMode;
        config.stateSize = 15;
        config.temporalWindow = 8;

        switch variant.paramMode
            case 'global'
                config.actionSize = 3;
            case '5subgroup'
                config.actionSize = 15;
            case 'per-particle'
                config.actionSize = popSize * 3;
            otherwise
                config.actionSize = 9;
        end
        config.targetEntropy = -config.actionSize;

        if strcmp(variant.stateMode, 'paper')
            config.stateMode = 'paper';
            config.rewardMode = 'paper';
            config.temporalWindow = 1;
            config.stateSize = popSize + 3;
        end

        config = applyParamMode(config);
        agent = AFSACPSO_Agent(config);
        evalc('agent.loadWeights(variant.weights);');
        stateEncoder = SimpleStateEncoder(config);
        [~, bestFit, ~] = psoMinimizeBenchmark(info.f, D, info.lb, info.ub, agent, stateEncoder, config);
        return;
    end

    loaded = load(variant.weights);
    preloadedAgent = loaded.trainedAgent;
    config = RLAMPSO_Config('pretrain');
    config.popSize = popSize;
    config.maxIterations = maxIter;
    config.verbose = false;
    config.trainDuringDeployment = false;
    fitFcn = @(x) info.f(x');
    [bestFit, ~, ~] = runRLAMPSOonBenchmark(fitFcn, D, info.lb(1), info.ub(1), info.fmin, config, preloadedAgent);
end

function writeSummaryCsv(filePath, summary)
    fid = fopen(filePath, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'Variant,Mean,Std,CpuTimeSec,Status\n');
    for i = 1:numel(summary)
        fprintf(fid, '%s,%.10g,%.10g,%.10g,%s\n', ...
            summary(i).name, summary(i).mean_all, summary(i).std_all, summary(i).cpu_time_sec, summary(i).status);
    end
end
