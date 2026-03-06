function run_apexpso_research_cycle(numRuns, maxIterations)
%RUN_APEXPSO_RESEARCH_CYCLE Evaluate edited APEXPSO variants only.
%
% This script:
%   1) Runs only APEXPSO variants (no baseline re-benchmarking).
%   2) Loads baseline statistics from "new result".
%   3) Produces direct mean/std fitness comparison tables.

    if nargin < 1 || isempty(numRuns)
        numRuns = getenvOrDefault('APEXPSO_RESEARCH_RUNS', 5);
    end
    if nargin < 2 || isempty(maxIterations)
        maxIterations = getenvOrDefault('APEXPSO_RESEARCH_MAX_ITERS', 1500);
    end

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    cd(repoRoot);

    addpath(repoRoot, '-begin');
    addpath(genpath(fullfile(repoRoot, 'algorithms')), '-begin');
    addpath(genpath(fullfile(repoRoot, 'shared')), '-begin');
    addpath(fullfile(repoRoot, 'data'), '-begin');
    rehash path;

    outputDir = resolveResearchOutputDir(repoRoot);
    ensureDir(outputDir);
    checkpointFile = fullfile(outputDir, 'apexpso_checkpoint.csv');
    ensureCheckpointFile(checkpointFile);
    completedKeys = loadCompletedKeyMap(checkpointFile);

    fprintf('==============================================\n');
    fprintf('  APEXPSO Research Cycle Runner\n');
    fprintf('==============================================\n');
    fprintf('Timestamp:      %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss')));
    fprintf('Runs/variant:   %d\n', numRuns);
    fprintf('Max iterations: %d\n', maxIterations);
    fprintf('Output dir:     %s\n\n', outputDir);

    oldQuietEnv = getenv('APEXPSO_CONFIG_QUIET');
    quietCleanup = onCleanup(@() setenv('APEXPSO_CONFIG_QUIET', oldQuietEnv)); %#ok<NASGU>
    setenv('APEXPSO_CONFIG_QUIET', '1');

    useParallel = getenvOrDefault('APEXPSO_RESEARCH_PARALLEL', 1) ~= 0;
    requestedWorkers = getenvOrDefault('APEXPSO_RESEARCH_WORKERS', 0);
    if useParallel && numRuns > 1
        if requestedWorkers > 0
            setenv('VIETANH_PARPOOL_WORKERS', num2str(requestedWorkers));
        end
        initializeParallelPool('parallel');
        pool = gcp('nocreate');
        if isempty(pool)
            warning('Parallel pool initialization failed; falling back to serial execution.');
            useParallel = false;
        else
            fprintf('Parallel mode: ON (%d workers)\n', pool.NumWorkers);
        end
    else
        useParallel = false;
        fprintf('Parallel mode: OFF\n');
    end

    useLOO = getenvOrDefault('APEXPSO_RESEARCH_LOO', 0) ~= 0;
    if useLOO
        variants = buildLOOVariants(maxIterations);
    else
        variants = buildVariants(maxIterations);
    end
    variantFilter = strtrim(getenv('APEXPSO_RESEARCH_VARIANTS'));
    if ~isempty(variantFilter)
        wanted = split(string(variantFilter), ",");
        wanted = strtrim(wanted);
        keep = false(numel(variants), 1);
        for i = 1:numel(variants)
            keep(i) = any(strcmp(string(variants(i).id), wanted)) || ...
                      any(strcmp(string(variants(i).name), wanted));
        end
        variants = variants(keep);
        if isempty(variants)
            error('APEXPSO_RESEARCH_VARIANTS filter selected no variants.');
        end
    end
    scenarios = [0, 5, 10];
    mapSize = [100, 100, 100];
    startPoint = [10, 95, 10];
    goalPoint = [97, 2, 10];

    numScenarios = numel(scenarios);
    numVariants = numel(variants);
    baselineScenario = strings(numScenarios, 1);
    baselineAlgo = strings(numScenarios, 1);
    baselineMean = zeros(numScenarios, 1);
    baselineStd = zeros(numScenarios, 1);
    for scenarioIdx = 1:numScenarios
        scenarioName = sprintf('Scenario%d', scenarioIdx);
        numDangerZones = scenarios(scenarioIdx);

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, scenarioIdx);
        dangerZones = generateDangerZones(numDangerZones, mapSize, terrainGrid, terrainX, terrainY);

        baseline = loadBaselineBestForScenario(repoRoot, scenarioIdx);
        baselineScenario(scenarioIdx) = scenarioName;
        baselineAlgo(scenarioIdx) = string(baseline.algorithm);
        baselineMean(scenarioIdx) = baseline.meanFitness;
        baselineStd(scenarioIdx) = baseline.stdFitness;

        fprintf('\n--- %s (%d danger zones) ---\n', scenarioName, numDangerZones);
        fprintf('Baseline best: %s | mean=%.4f | std=%.4f\n', ...
            baseline.algorithm, baseline.meanFitness, baseline.stdFitness);

        baseSeeds = zeros(numRuns, 1);
        for run = 1:numRuns
            baseSeeds(run) = run * 1000 + scenarioIdx * 100;
        end
        fairSeeds = getenvOrDefault('APEXPSO_RESEARCH_FAIR_SEEDS', 1) ~= 0;

        for v = 1:numVariants
            variant = variants(v);
            fprintf('  Running variant: %s\n', variant.name);
            seeds = zeros(numRuns, 1);
            if fairSeeds
                seeds = baseSeeds;
            else
                for run = 1:numRuns
                    seeds(run) = run * 1000 + scenarioIdx * 100 + v * 10;
                end
            end

            pendingRuns = [];
            for run = 1:numRuns
                if ~isKey(completedKeys, makeRunKey(scenarioName, variant.name, run))
                    pendingRuns(end+1, 1) = run; %#ok<AGROW>
                end
            end

            if isempty(pendingRuns)
                fprintf('    Resume: all %d runs already complete, skipping.\n', numRuns);
                continue;
            end

            if useParallel
                progressQueue = parallel.pool.DataQueue;
                afterEach(progressQueue, @checkpointRunResult);
                parfor idx = 1:numel(pendingRuns)
                    run = pendingRuns(idx);
                    fitnessValue = executeSingleRun( ...
                        seeds(run), maxIterations, variant, ...
                        startPoint, goalPoint, dangerZones, ...
                        terrainGrid, terrainX, terrainY, mapSize);
                    send(progressQueue, struct( ...
                        'Scenario', scenarioName, ...
                        'Variant', variant.name, ...
                        'Run', run, ...
                        'Seed', seeds(run), ...
                        'Fitness', fitnessValue));
                end
            else
                for idx = 1:numel(pendingRuns)
                    run = pendingRuns(idx);
                    fitnessValue = executeSingleRun( ...
                        seeds(run), maxIterations, variant, ...
                        startPoint, goalPoint, dangerZones, ...
                        terrainGrid, terrainX, terrainY, mapSize);
                    checkpointRunResult(struct( ...
                        'Scenario', scenarioName, ...
                        'Variant', variant.name, ...
                        'Run', run, ...
                        'Seed', seeds(run), ...
                        'Fitness', fitnessValue));
                end
            end
        end
    end

    allRuns = readtable(checkpointFile, 'VariableNamingRule', 'preserve');
    allRuns.Scenario = string(allRuns.Scenario);
    allRuns.Variant = string(allRuns.Variant);
    allRuns = sortrows(allRuns, {'Scenario', 'Variant', 'Run'}, {'ascend', 'ascend', 'ascend'});

    summary = summarizeRuns(allRuns, variants, baselineScenario, baselineAlgo, baselineMean, baselineStd);
    baselineRef = table(baselineScenario, baselineAlgo, baselineMean, baselineStd, ...
        'VariableNames', {'Scenario', 'BaselineBestAlgorithm', 'BaselineMeanFitness', 'BaselineStdFitness'});

    writetable(allRuns, fullfile(outputDir, 'apexpso_all_runs.csv'));
    writetable(summary, fullfile(outputDir, 'apexpso_summary_vs_baseline.csv'));
    writetable(baselineRef, fullfile(outputDir, 'baseline_reference.csv'));
    writeMarkdownSummary(summary, fullfile(outputDir, 'comparison.md'));

    fprintf('\nSaved files:\n');
    fprintf('  %s\n', fullfile(outputDir, 'apexpso_all_runs.csv'));
    fprintf('  %s\n', fullfile(outputDir, 'apexpso_summary_vs_baseline.csv'));
    fprintf('  %s\n', fullfile(outputDir, 'baseline_reference.csv'));
    fprintf('  %s\n', fullfile(outputDir, 'comparison.md'));

    function checkpointRunResult(runData)
        key = makeRunKey(runData.Scenario, runData.Variant, runData.Run);
        if isKey(completedKeys, key)
            return;
        end
        appendCheckpointRow(checkpointFile, runData);
        completedKeys(key) = true;
        fprintf('    Run %2d/%2d | seed=%d | fitness=%.4f\n', ...
            runData.Run, numRuns, runData.Seed, runData.Fitness);
    end
end

function variants = buildVariants(maxIterations)
    baseOverrides = struct();
    baseOverrides.disableVisualization = true;
    baseOverrides.maxIterations = maxIterations;
    baseOverrides.useCrossScaleState = true;
    baseOverrides.useTransformerState = false;
    baseOverrides.useRankResidualControl = true;
    % Post-LOO default stack: keep only the components that remained defensible
    % after the 50-run leave-one-out study.
    baseOverrides.useActorUncertaintyPenalty = true;
    baseOverrides.uncertaintyPenaltyWeight = 0.08;
    baseOverrides.useCuriosityBonus = true;
    baseOverrides.useEntropyUncertaintyCoupling = false;
    baseOverrides.useHuberCriticLoss = true;
    baseOverrides.useOverestimationPenalty = true;

    variants = repmat(struct('name', '', 'id', '', 'paramMode', '', 'overrides', struct()), 2, 1);

    variants(1).name = 'APEX-V7-LOOReduced-Base';
    variants(1).id = 'v7_loo_reduced_base';
    variants(1).paramMode = 'rank-residual';
    variants(1).overrides = baseOverrides;

    variants(2).name = 'APEX-V7-WithEntropyUncertainty';
    variants(2).id = 'v7_with_entropy_uncertainty';
    variants(2).paramMode = 'rank-residual';
    variants(2).overrides = mergeStruct(baseOverrides, struct('useEntropyUncertaintyCoupling', true));
end

function variants = buildLOOVariants(maxIterations)
    base = struct();
    base.disableVisualization = true;
    base.maxIterations = maxIterations;
    base.useCrossScaleState = true;
    base.useTransformerState = false;
    base.useRankResidualControl = true;
    base.useActorUncertaintyPenalty = true;
    base.uncertaintyPenaltyWeight = 0.08;
    base.useCuriosityBonus = true;
    base.useEntropyUncertaintyCoupling = false;
    base.useHuberCriticLoss = true;
    base.useOverestimationPenalty = true;

    variants = repmat(struct('name', '', 'id', '', 'paramMode', '', 'overrides', struct()), 3, 1);

    variants(1).name = 'APEX-LOO-Full';
    variants(1).id = 'loo_full';
    variants(1).paramMode = 'rank-residual';
    variants(1).overrides = base;

    variants(2).name = 'APEX-LOO-NoRankResidual';
    variants(2).id = 'loo_no_rank_residual';
    variants(2).paramMode = 'global';
    variants(2).overrides = mergeStruct(base, struct('useRankResidualControl', false));

    variants(3).name = 'APEX-LOO-WithEntropyUncertainty';
    variants(3).id = 'loo_with_entropy_uncertainty';
    variants(3).paramMode = 'rank-residual';
    variants(3).overrides = mergeStruct(base, struct('useEntropyUncertaintyCoupling', true));
end

function baseline = loadBaselineBestForScenario(repoRoot, scenarioIdx)
    rlDir = fullfile(repoRoot, 'new result', 'rl_based');
    othersDir = fullfile(repoRoot, 'new result', 'others_results');

    rlFile = resolveLatestSummary(rlDir, scenarioIdx);
    othersFile = resolveLatestSummary(othersDir, scenarioIdx);

    t1 = readtable(rlFile, 'VariableNamingRule', 'preserve');
    t2 = readtable(othersFile, 'VariableNamingRule', 'preserve');
    combined = [t1(:, {'Algorithm', 'Mean_GlobalFitness', 'Std_GlobalFitness'}); ...
                t2(:, {'Algorithm', 'Mean_GlobalFitness', 'Std_GlobalFitness'})];

    [bestMean, bestIdx] = min(combined.Mean_GlobalFitness);
    baseline = struct();
    baseline.algorithm = combined.Algorithm{bestIdx};
    baseline.meanFitness = bestMean;
    baseline.stdFitness = combined.Std_GlobalFitness(bestIdx);
    baseline.sourceFiles = {rlFile, othersFile}; %#ok<STRNU>
end

function fitnessValue = executeSingleRun(seed, maxIterations, variant, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize)
    rng(seed, 'twister');

    params = struct();
    params.maxIterations = maxIterations;
    params.paramMode = variant.paramMode;
    params.variant = variant.id;
    params.configOverrides = variant.overrides;

    [~, stats] = callGlobalPlanningAlgorithm('APEXPSO_Online', ...
        startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params);

    if isfield(stats, 'actualBestFitness')
        fitnessValue = stats.actualBestFitness;
    elseif isfield(stats, 'finalFitness')
        fitnessValue = stats.finalFitness;
    else
        error('Missing fitness in algorithm stats.');
    end
end

function summary = summarizeRuns(allRuns, variants, baselineScenario, baselineAlgo, baselineMean, baselineStd)
    numScenarios = numel(baselineScenario);
    numVariants = numel(variants);
    numRows = numScenarios * numVariants;

    scenarioCol = strings(numRows, 1);
    variantCol = strings(numRows, 1);
    nCol = zeros(numRows, 1);
    meanCol = zeros(numRows, 1);
    stdCol = zeros(numRows, 1);
    baselineAlgoCol = strings(numRows, 1);
    baselineMeanCol = zeros(numRows, 1);
    baselineStdCol = zeros(numRows, 1);
    deltaCol = zeros(numRows, 1);
    isTop1Col = false(numRows, 1);

    idx = 0;
    for s = 1:numScenarios
        scenarioName = baselineScenario(s);
        baselineBest = baselineMean(s);
        for v = 1:numVariants
            idx = idx + 1;
            variantName = string(variants(v).name);
            mask = allRuns.Scenario == scenarioName & allRuns.Variant == variantName;
            vals = allRuns.Fitness(mask);

            scenarioCol(idx) = scenarioName;
            variantCol(idx) = variantName;
            nCol(idx) = numel(vals);
            meanCol(idx) = mean(vals);
            stdCol(idx) = std(vals);
            baselineAlgoCol(idx) = baselineAlgo(s);
            baselineMeanCol(idx) = baselineMean(s);
            baselineStdCol(idx) = baselineStd(s);
            deltaCol(idx) = meanCol(idx) - baselineBest;
            isTop1Col(idx) = meanCol(idx) < baselineBest;
        end
    end

    summary = table(scenarioCol, variantCol, nCol, meanCol, stdCol, ...
        baselineAlgoCol, baselineMeanCol, baselineStdCol, deltaCol, isTop1Col, ...
        'VariableNames', {'Scenario', 'Variant', 'NumRuns', 'MeanFitness', 'StdFitness', ...
        'BaselineBestAlgorithm', 'BaselineBestMeanFitness', 'BaselineBestStdFitness', ...
        'DeltaVsBaselineBest', 'IsTop1VsStoredBaselines'});

    summary = sortrows(summary, {'Scenario', 'MeanFitness'}, {'ascend', 'ascend'});
end

function writeMarkdownSummary(summary, outputPath)
    fid = fopen(outputPath, 'w');
    if fid == -1
        error('Failed to open markdown output: %s', outputPath);
    end
    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, '# APEXPSO Research Cycle Summary\n\n');
    fprintf(fid, '| Scenario | Variant | Runs | Mean Fitness | Std | Baseline Best | Baseline Mean | Delta | Top-1 |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---|---:|---:|---|\n');
    for i = 1:height(summary)
        fprintf(fid, '| %s | %s | %d | %.4f | %.4f | %s | %.4f | %.4f | %s |\n', ...
            summary.Scenario(i), summary.Variant(i), summary.NumRuns(i), ...
            summary.MeanFitness(i), summary.StdFitness(i), ...
            summary.BaselineBestAlgorithm(i), summary.BaselineBestMeanFitness(i), ...
            summary.DeltaVsBaselineBest(i), boolToYesNo(summary.IsTop1VsStoredBaselines(i)));
    end
end

function filePath = resolveLatestSummary(baseDir, scenarioIdx)
    pattern = sprintf('TTest_AlgorithmSummary_Scenario%d_*.csv', scenarioIdx);
    files = dir(fullfile(baseDir, pattern));
    if isempty(files)
        error('No baseline summary file found: %s', fullfile(baseDir, pattern));
    end
    [~, bestIdx] = max([files.datenum]);
    filePath = fullfile(files(bestIdx).folder, files(bestIdx).name);
end

function out = mergeStruct(base, updates)
    out = base;
    f = fieldnames(updates);
    for i = 1:numel(f)
        out.(f{i}) = updates.(f{i});
    end
end

function ensureDir(pathStr)
    if exist(pathStr, 'dir') ~= 7
        mkdir(pathStr);
    end
end

function outputDir = resolveResearchOutputDir(repoRoot)
    resumeDir = strtrim(getenv('APEXPSO_RESEARCH_RESUME_DIR'));
    if ~isempty(resumeDir)
        outputDir = resumeDir;
        return;
    end

    outputOverride = strtrim(getenv('APEXPSO_RESEARCH_OUTPUT_DIR'));
    if ~isempty(outputOverride)
        outputDir = outputOverride;
        return;
    end

    runStamp = char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
    outputDir = fullfile(repoRoot, 'outputs', 'research', runStamp);
end

function ensureCheckpointFile(checkpointFile)
    if exist(checkpointFile, 'file') == 2
        return;
    end

    fid = fopen(checkpointFile, 'w');
    if fid == -1
        error('Failed to create checkpoint file: %s', checkpointFile);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'Scenario,Variant,Run,Seed,Fitness\n');
end

function completedKeys = loadCompletedKeyMap(checkpointFile)
    completedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    if exist(checkpointFile, 'file') ~= 2
        return;
    end

    checkpointTable = readtable(checkpointFile, 'VariableNamingRule', 'preserve');
    for i = 1:height(checkpointTable)
        completedKeys(makeRunKey(checkpointTable.Scenario(i), checkpointTable.Variant(i), checkpointTable.Run(i))) = true;
    end
end

function appendCheckpointRow(checkpointFile, runData)
    fid = fopen(checkpointFile, 'a');
    if fid == -1
        error('Failed to append checkpoint file: %s', checkpointFile);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s,%s,%d,%d,%.10f\n', ...
        char(string(runData.Scenario)), char(string(runData.Variant)), ...
        runData.Run, runData.Seed, runData.Fitness);
end

function key = makeRunKey(scenarioName, variantName, runNumber)
    key = sprintf('%s|%s|%d', char(string(scenarioName)), char(string(variantName)), runNumber);
end

function v = getenvOrDefault(envName, defaultVal)
    raw = strtrim(getenv(envName));
    if isempty(raw)
        v = defaultVal;
        return;
    end
    parsed = str2double(raw);
    if isnan(parsed)
        v = defaultVal;
    else
        v = parsed;
    end
end

function out = boolToYesNo(flag)
    if flag
        out = 'YES';
    else
        out = 'NO';
    end
end
