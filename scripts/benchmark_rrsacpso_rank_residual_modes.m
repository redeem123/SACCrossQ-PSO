function benchmark_rrsacpso_rank_residual_modes(numRuns, maxIterations, popSize, scenarios, numWorkers)
%BENCHMARK_APEXPSO_RANK_RESIDUAL_MODES
% Compare RRSACPSO parameterization modes under the same SAC backbone.
%
% The intended comparison is:
%   - rank-residual
%   - global
%   - 5subgroup
%   - per-particle
%
% To isolate the action-parameterization question, critic-side research
% branches are disabled via config overrides.

    if nargin < 1 || isempty(numRuns)
        numRuns = 8;
    end
    if nargin < 2 || isempty(maxIterations)
        maxIterations = 600;
    end
    if nargin < 3 || isempty(popSize)
        popSize = 40;
    end
    if nargin < 4 || isempty(scenarios)
        scenarios = [0; 5; 10];
    end
    if nargin < 5 || isempty(numWorkers)
        numWorkers = 0;
    end

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(fullfile(repoRoot, 'data'));

    useParallel = numWorkers > 1;
    oldParpoolWorkers = getenv('VIETANH_PARPOOL_WORKERS');
    parpoolCleanup = onCleanup(@() setenv('VIETANH_PARPOOL_WORKERS', oldParpoolWorkers));
    if useParallel
        setenv('VIETANH_PARPOOL_WORKERS', num2str(numWorkers));
        initializeParallelPool('parallel');
    end

    outputDir = fullfile(repoRoot, 'outputs', 'research', ...
        sprintf('rrsacpso_rank_residual_modes_%s', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
    resultsDir = fullfile(outputDir, 'results');
    if exist(resultsDir, 'dir') ~= 7
        mkdir(resultsDir);
    end

    oldResultsDir = getenv('VIETANH_RESULTS_DIR');
    resultsCleanup = onCleanup(@() setenv('VIETANH_RESULTS_DIR', oldResultsDir));
    setenv('VIETANH_RESULTS_DIR', resultsDir);

    oldQuiet = getenv('APEXPSO_CONFIG_QUIET');
    quietCleanup = onCleanup(@() setenv('APEXPSO_CONFIG_QUIET', oldQuiet));
    setenv('APEXPSO_CONFIG_QUIET', '1');

    environment = resolveTerrainEnvironment(defaultEnvironment());
    variants = buildVariants();
    baseOverrides = buildBaseOverrides(popSize);

    fprintf('============================================================\n');
    fprintf('RRSACPSO RankResidualControl Benchmark\n');
    fprintf('============================================================\n');
    fprintf('Runs: %d\n', numRuns);
    fprintf('Max iterations: %d\n', maxIterations);
    fprintf('Population size: %d\n', popSize);
    fprintf('Workers: %d\n', max(1, numWorkers));
    fprintf('Scenarios: %s\n', mat2str(scenarios(:)'));
    fprintf('Results: %s\n\n', outputDir);

    overviewRows = {};
    for scenarioIdx = 1:numel(scenarios)
        numDangerZones = scenarios(scenarioIdx);
        fprintf('\n============================================================\n');
        fprintf('Scenario %d (%d danger zones)\n', scenarioIdx, numDangerZones);
        fprintf('============================================================\n');

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
            environment.mapSize, scenarioIdx, environment.terrainFile);
        dangerZones = generateDangerZones(numDangerZones, environment.mapSize, terrainGrid, terrainX, terrainY);

        runResults = executeScenarioRuns(numRuns, variants, environment, dangerZones, ...
            terrainGrid, terrainX, terrainY, maxIterations, baseOverrides, scenarioIdx, useParallel);

        performTTest(runResults, variants, scenarioIdx);
        generateReports(runResults, variants, scenarioIdx);

        save(fullfile(outputDir, sprintf('scenario%d_run_results.mat', scenarioIdx)), ...
            'runResults', 'scenarioIdx', 'numDangerZones', 'variants', 'numRuns', 'maxIterations', 'popSize');

        scenarioResults = load(fullfile(resultsDir, sprintf('TTest_Results_Scenario%d.mat', scenarioIdx)));
        writeScenarioMarkdown(outputDir, scenarioIdx, numDangerZones, scenarioResults);
        overviewRows = [overviewRows; extractOverviewRows(scenarioIdx, numDangerZones, scenarioResults)]; %#ok<AGROW>
    end

    writeOverviewCSV(outputDir, overviewRows);
    fprintf('\nBenchmark complete. Outputs saved to:\n  %s\n', outputDir);

    clear resultsCleanup quietCleanup parpoolCleanup;
end

function environment = defaultEnvironment()
    environment = struct( ...
        'mapSize', [100, 100, 100], ...
        'startPoint', [10, 95, 10], ...
        'goalPoint', [97, 2, 10], ...
        'terrainFile', '');
end

function variants = buildVariants()
    variants = {
        makeVariant('RRSACPSO (RankResidualControl)', 'RRSACPSO_RankResidual', 'rank-residual', 1)
        makeVariant('RRSACPSO (Global)', 'RRSACPSO_Global', 'global', 2)
        makeVariant('RRSACPSO (5-Subgroup)', 'RRSACPSO_5Subgroup', '5subgroup', 3)
        makeVariant('RRSACPSO (Per-Particle)', 'RRSACPSO_PerParticle', 'per-particle', 4)
    };
end

function variant = makeVariant(displayName, fieldName, paramMode, seedOffset)
    variant = struct();
    variant.function = @(varargin) directGlobalPlanning(varargin{:});
    variant.category = 'RRSACPSO-ParamModes';
    variant.displayName = displayName;
    variant.fieldName = fieldName;
    variant.paramMode = paramMode;
    variant.seedOffset = seedOffset;
    variant.requiresSerial = true;
end

function overrides = buildBaseOverrides(popSize)
    overrides = struct();
    overrides.popSize = popSize;
    overrides.useTQCCritic = false;
    overrides.useCrossQCritic = false;
    overrides.useREDQCritic = false;
    overrides.useAQECritic = false;
    overrides.useDroQCritic = false;
    overrides.useCrossScaleState = false;
    overrides.useCrossScaleBranching = false;
    overrides.useCrossScaleGatedFusion = false;
    overrides.useSimBaBackbone = false;
    overrides.disableVisualization = true;
end

function runResults = runVariantBatch(variants, seedBase, environment, dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides)
    runResults = struct();
    startPoint = environment.startPoint;
    goalPoint = environment.goalPoint;
    mapSize = environment.mapSize;
    for idx = 1:numel(variants)
        variant = variants{idx};
        rng(seedBase + variant.seedOffset);
        params = struct( ...
            'maxIterations', maxIterations, ...
            'paramMode', variant.paramMode, ...
            'configOverrides', baseOverrides);

        stepTimer = tic;
        [~, globalPath, algorithmSpecificStats] = evalc( ...
            'callApexVariant(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params)');
        executionTime = toc(stepTimer);

        metrics = buildMetrics(globalPath, algorithmSpecificStats, executionTime, variant.displayName);
        runResults.(variant.fieldName) = metrics;
        runResults.([variant.fieldName '_finalPath']) = globalPath;
        runResults.([variant.fieldName '_globalPath']) = metrics.finalGlobalPath;
    end
end

function [globalPath, algorithmSpecificStats] = callApexVariant(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params)
    [globalPath, algorithmSpecificStats] = callGlobalPlanningAlgorithm( ...
        'RRSACPSO_Online', startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params);
end

function runResults = executeScenarioRuns(numRuns, variants, environment, dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides, scenarioIdx, useParallel)
    runResults = cell(numRuns, 1);
    if useParallel
        parfor runIdx = 1:numRuns
            seedBase = runIdx * 1000 + scenarioIdx * 100;
            runResults{runIdx} = runVariantBatch(variants, seedBase, environment, ...
                dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides);
        end
        fprintf('  Completed %d/%d runs in parallel.\n', numRuns, numRuns);
    else
        for runIdx = 1:numRuns
            seedBase = runIdx * 1000 + scenarioIdx * 100;
            runResults{runIdx} = runVariantBatch(variants, seedBase, environment, ...
                dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides);
            fprintf('  Completed run %d/%d\n', runIdx, numRuns);
        end
    end
end

function metrics = buildMetrics(globalPath, algorithmSpecificStats, executionTime, displayName)
    metrics = struct();
    metrics.algorithmName = displayName;
    metrics.executionTime = executionTime;
    metrics.pathLength = calculatePathLength(globalPath);
    metrics.finalGlobalPath = globalPath;

    if isfield(algorithmSpecificStats, 'actualBestFitness')
        metrics.actualBestFitness = algorithmSpecificStats.actualBestFitness;
    end
    if isfield(algorithmSpecificStats, 'fitnessComponents')
        metrics.fitnessComponents = algorithmSpecificStats.fitnessComponents;
    end
    if isfield(algorithmSpecificStats, 'convergenceHistory')
        metrics.convergenceHistory = algorithmSpecificStats.convergenceHistory;
    end
    if isfield(algorithmSpecificStats, 'rewardHistory')
        metrics.rewardHistory = algorithmSpecificStats.rewardHistory;
    end
    if isfield(algorithmSpecificStats, 'criticLossHistory')
        metrics.criticLossHistory = algorithmSpecificStats.criticLossHistory;
    end
    if isfield(algorithmSpecificStats, 'parameterHistory')
        paramHistory = algorithmSpecificStats.parameterHistory;
        if isfield(paramHistory, 'w'), metrics.w_history = paramHistory.w; end
        if isfield(paramHistory, 'c1'), metrics.c1_history = paramHistory.c1; end
        if isfield(paramHistory, 'c2'), metrics.c2_history = paramHistory.c2; end
        if isfield(paramHistory, 'w_samples'), metrics.w_samples = paramHistory.w_samples; end
        if isfield(paramHistory, 'c1_samples'), metrics.c1_samples = paramHistory.c1_samples; end
        if isfield(paramHistory, 'c2_samples'), metrics.c2_samples = paramHistory.c2_samples; end
    end
end

function rows = extractOverviewRows(scenarioIdx, numDangerZones, scenarioResults)
    rows = {};
    names = scenarioResults.algorithmNames;
    pairwise = scenarioResults.pairwiseResults;
    rrIdx = find(strcmp(names, 'RRSACPSO (RankResidualControl)'), 1);
    if isempty(rrIdx)
        return;
    end

    for idx = 1:numel(pairwise)
        row = pairwise(idx);
        if strcmp(row.algorithm_1, 'RRSACPSO (RankResidualControl)')
            opponent = row.algorithm_2;
            rrMean = row.mean_1;
            oppMean = row.mean_2;
            meanDiff = row.mean_diff;
        elseif strcmp(row.algorithm_2, 'RRSACPSO (RankResidualControl)')
            opponent = row.algorithm_1;
            rrMean = row.mean_2;
            oppMean = row.mean_1;
            meanDiff = -row.mean_diff;
        else
            continue;
        end

        rows(end + 1, :) = { ... %#ok<AGROW>
            scenarioIdx, ...
            numDangerZones, ...
            opponent, ...
            rrMean, ...
            oppMean, ...
            meanDiff, ...
            row.p_raw_t, ...
            row.p_holm_t, ...
            row.p_raw_wilcoxon, ...
            row.p_holm_wilcoxon, ...
            row.rank_biserial_r ...
        };
    end
end

function writeOverviewCSV(outputDir, rows)
    filename = fullfile(outputDir, 'rank_residual_vs_other_modes.csv');
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open overview CSV for writing: %s', filename);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, ['scenario_id,num_danger_zones,opponent,mean_rank_residual,mean_opponent,', ...
        'mean_diff_rank_minus_opponent,p_raw_t,p_holm_t,p_raw_wilcoxon,p_holm_wilcoxon,rank_biserial_r\n']);
    for idx = 1:size(rows, 1)
        row = rows(idx, :);
        fprintf(fid, '%d,%d,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n', ...
            row{1}, row{2}, csvSafe(row{3}), row{4}, row{5}, row{6}, row{7}, row{8}, row{9}, row{10}, row{11});
    end

    clear cleanup;
end

function writeScenarioMarkdown(outputDir, scenarioIdx, numDangerZones, scenarioResults)
    summary = scenarioResults.summaryResults;
    pairwise = scenarioResults.pairwiseResults;
    filename = fullfile(outputDir, sprintf('scenario%d_summary.md', scenarioIdx));
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open markdown summary for writing: %s', filename);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, '# Scenario %d\n\n', scenarioIdx);
    fprintf(fid, '- Danger zones: %d\n', numDangerZones);
    fprintf(fid, '- Metric: %s\n\n', scenarioResults.metricName);

    fprintf(fid, '## Means\n\n');
    fprintf(fid, '| Algorithm | Mean | Std | Median |\n');
    fprintf(fid, '|---|---:|---:|---:|\n');
    for idx = 1:numel(summary)
        row = summary(idx);
        fprintf(fid, '| %s | %.4f | %.4f | %.4f |\n', row.algorithm, row.mean, row.std, row.median);
    end

    fprintf(fid, '\n## RankResidualControl Pairwise\n\n');
    fprintf(fid, '| Opponent | Mean diff (RR - Opp) | p raw t | p raw Wilcoxon | p Holm Wilcoxon |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|\n');
    for idx = 1:numel(pairwise)
        row = pairwise(idx);
        if strcmp(row.algorithm_1, 'RRSACPSO (RankResidualControl)')
            opponent = row.algorithm_2;
            meanDiff = row.mean_diff;
        elseif strcmp(row.algorithm_2, 'RRSACPSO (RankResidualControl)')
            opponent = row.algorithm_1;
            meanDiff = -row.mean_diff;
        else
            continue;
        end
        fprintf(fid, '| %s | %.4f | %.6f | %.6f | %.6f |\n', ...
            opponent, meanDiff, row.p_raw_t, row.p_raw_wilcoxon, row.p_holm_wilcoxon);
    end

    clear cleanup;
end

function textOut = csvSafe(textIn)
    if isempty(textIn)
        textOut = '';
        return;
    end
    textOut = strrep(textIn, ',', ' ');
end
