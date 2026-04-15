function benchmark_afsacpso_attractor_field_modes(numRuns, maxIterations, popSize, scenarios, numWorkers)
%BENCHMARK_AFSACPSO_ATTRACTOR_FIELD_MODES
% Compare AFSACPSO parameterization modes under the same SAC backbone.
%
% The intended comparison is:
%   - attractor-field
%   - global
%   - 5subgroup
%   - per-particle
%
% To isolate the action-parameterization question, critic-side research
% branches are disabled via config overrides. The retained comparison uses
% frozen pretrained checkpoints for all four control modes.

    if nargin < 1 || isempty(numRuns)
        numRuns = 8;
    end
    if nargin < 2 || isempty(maxIterations)
        maxIterations = 600;
    end
    if nargin < 3 || isempty(popSize)
        popSize = 40;
    end
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    dataDir = fullfile(repoRoot, 'data');

    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(dataDir);

    if nargin < 4 || isempty(scenarios)
        scenarios = buildDefaultScenarioSet(dataDir);
    end
    if nargin < 5 || isempty(numWorkers)
        numWorkers = 0;
    end

    useParallel = numWorkers > 1;
    oldParpoolWorkers = getenv('VIETANH_PARPOOL_WORKERS');
    parpoolCleanup = onCleanup(@() setenv('VIETANH_PARPOOL_WORKERS', oldParpoolWorkers));
    if useParallel
        setenv('VIETANH_PARPOOL_WORKERS', num2str(numWorkers));
        initializeParallelPool('parallel');
    end

    outputDir = fullfile(repoRoot, 'outputs', 'research', ...
        sprintf('afsacpso_attractor_field_modes_%s', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
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

    baseEnvironment = defaultEnvironment(dataDir);
    scenarioSet = normalizeScenarioSet(scenarios, dataDir);
    variants = buildVariants(repoRoot);
    baseOverrides = buildBaseOverrides(popSize);

    fprintf('============================================================\n');
    fprintf('AFSACPSO Attractor-Field Benchmark\n');
    fprintf('============================================================\n');
    fprintf('Runs: %d\n', numRuns);
    fprintf('Max iterations: %d\n', maxIterations);
    fprintf('Population size: %d\n', popSize);
    fprintf('Workers: %d\n', max(1, numWorkers));
    fprintf('Scenarios: %s\n', formatScenarioSummary(scenarioSet));
    fprintf('Results: %s\n\n', outputDir);

    overviewRows = {};
    for scenarioIdx = 1:numel(scenarioSet)
        [scenarioDefinition, scenarioEnvironment] = resolveScenarioDefinition( ...
            scenarioSet, scenarioIdx, baseEnvironment);
        numDangerZones = scenarioDefinition.numDangerZones;
        fprintf('\n============================================================\n');
        fprintf('Scenario %d: %s\n', scenarioIdx, scenarioDefinition.label);
        fprintf('Danger zones: %d\n', numDangerZones);
        fprintf('Terrain: %s\n', scenarioDefinition.terrainFile);
        fprintf('============================================================\n');

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
            scenarioEnvironment.mapSize, scenarioIdx, scenarioEnvironment.terrainFile);
        dangerZones = generateDangerZones(numDangerZones, scenarioEnvironment.mapSize, terrainGrid, terrainX, terrainY);

        runResults = executeScenarioRuns(numRuns, variants, scenarioEnvironment, dangerZones, ...
            terrainGrid, terrainX, terrainY, maxIterations, baseOverrides, scenarioIdx, useParallel);

        performTTest(runResults, variants, scenarioIdx);
        generateReports(runResults, variants, scenarioIdx);

        save(fullfile(outputDir, sprintf('scenario%d_run_results.mat', scenarioIdx)), ...
            'runResults', 'scenarioIdx', 'scenarioDefinition', 'scenarioEnvironment', ...
            'numDangerZones', 'variants', 'numRuns', 'maxIterations', 'popSize');

        scenarioResults = load(fullfile(resultsDir, sprintf('TTest_Results_Scenario%d.mat', scenarioIdx)));
        writeScenarioMarkdown(outputDir, scenarioIdx, scenarioDefinition, scenarioResults);
        overviewRows = [overviewRows; extractOverviewRows(scenarioIdx, scenarioDefinition, scenarioResults)]; %#ok<AGROW>
    end

    writeOverviewCSV(outputDir, overviewRows);
    fprintf('\nBenchmark complete. Outputs saved to:\n  %s\n', outputDir);

    clear resultsCleanup quietCleanup parpoolCleanup;
end

function environment = defaultEnvironment(dataDir)
    environment = struct( ...
        'mapSize', [100, 100, 100], ...
        'startPoint', [10, 95, 10], ...
        'goalPoint', [97, 2, 10], ...
        'terrainFile', fullfile(dataDir, 'ChrismasTerrain2.tif'));
end

function variants = buildVariants(repoRoot)
    modelDir = fullfile(repoRoot, 'models', 'AFSACPSO');
    variants = {
        makeVariant('AFSACPSO', 'AFSACPSO', 'attractor-field', 1, ...
            fullfile(modelDir, 'pretrained_sac_attractor.mat'))
        makeVariant('SACPSO-Global', 'SACPSO_Global', 'global', 2, ...
            fullfile(modelDir, 'pretrained_sac_global.mat'))
        makeVariant('SACPSO-5Subgroup', 'SACPSO_5Subgroup', '5subgroup', 3, ...
            fullfile(modelDir, 'pretrained_sac_benchmarks_5subgroup.mat'))
        makeVariant('SACPSO-PerParticle', 'SACPSO_PerParticle', 'per-particle', 4, ...
            fullfile(modelDir, 'pretrained_sac_perparticle.mat'))
    };
end

function variant = makeVariant(displayName, fieldName, paramMode, seedOffset, checkpointPath)
    variant = struct();
    variant.function = @(varargin) directGlobalPlanning(varargin{:});
    variant.category = 'AFSACPSO-ParamModes';
    variant.displayName = displayName;
    variant.fieldName = fieldName;
    variant.paramMode = paramMode;
    variant.seedOffset = seedOffset;
    variant.checkpointPath = checkpointPath;
    variant.requiresSerial = true;
end

function overrides = buildBaseOverrides(popSize)
    overrides = struct();
    overrides.popSize = popSize;
    overrides.stateSize = 15;
    overrides.useAttention = false;
    overrides.useSAC = true;
    overrides.useTargetNetworks = true;
    overrides.useTQCCritic = false;
    overrides.useCrossQCritic = false;
    overrides.useREDQCritic = false;
    overrides.useAQECritic = false;
    overrides.useDroQCritic = false;
    overrides.useCrossScaleState = false;
    overrides.useCrossScaleBranching = false;
    overrides.useCrossScaleGatedFusion = false;
    overrides.useSimBaBackbone = false;
    overrides.useObservationNormalization = false;
    overrides.usePrioritizedReplay = false;
    overrides.useResidualCriticDecomposition = false;
    overrides.usePilarReturns = false;
    overrides.useDelayedPolicyUpdates = false;
    overrides.useCriticBatchNorm = false;
    overrides.useActorBatchNorm = false;
    overrides.useJointCriticBatchForBN = false;
    overrides.useWeightNormCritic = false;
    overrides.numCritics = 2;
    overrides.utdRatio = 1;
    overrides.gradientStepsPerTraining = 1;
    overrides.disableVisualization = true;
    overrides.useTrajectoryPolish = false;
end

function runResults = runVariantBatch(variants, seedBase, environment, dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides)
    runResults = struct();
    startPoint = environment.startPoint;
    goalPoint = environment.goalPoint;
    mapSize = environment.mapSize;
    for idx = 1:numel(variants)
        variant = variants{idx};
        rng(seedBase + variant.seedOffset);
        variantOverrides = baseOverrides;
        variantOverrides.pretrainedWeightsPath = variant.checkpointPath;
        variantOverrides.warmupPeriod = maxIterations + 1;
        params = struct( ...
            'maxIterations', maxIterations, ...
            'paramMode', variant.paramMode, ...
            'configOverrides', variantOverrides);

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
        'AFSACPSO_Online', startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params);
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
    metrics.globalPlanTimes = 0;
    metrics.globalPlanDurations = executionTime;
    metrics.numGlobalPlans = 1;
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

function rows = extractOverviewRows(scenarioIdx, scenarioDefinition, scenarioResults)
    rows = {};
    names = scenarioResults.algorithmNames;
    pairwise = scenarioResults.pairwiseResults;
    rrIdx = find(strcmp(names, 'AFSACPSO'), 1);
    if isempty(rrIdx)
        return;
    end

    for idx = 1:numel(pairwise)
        row = pairwise(idx);
        if strcmp(row.algorithm_1, 'AFSACPSO')
            opponent = row.algorithm_2;
            rrMean = row.mean_1;
            oppMean = row.mean_2;
            meanDiff = row.mean_diff;
        elseif strcmp(row.algorithm_2, 'AFSACPSO')
            opponent = row.algorithm_1;
            rrMean = row.mean_2;
            oppMean = row.mean_1;
            meanDiff = -row.mean_diff;
        else
            continue;
        end

        rows(end + 1, :) = { ... %#ok<AGROW>
            scenarioIdx, ...
            scenarioDefinition.label, ...
            scenarioDefinition.numDangerZones, ...
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
    filename = fullfile(outputDir, 'attractor_field_vs_other_modes.csv');
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open overview CSV for writing: %s', filename);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, ['scenario_id,scenario_label,num_danger_zones,opponent,mean_attractor_field,mean_opponent,', ...
        'mean_diff_rank_minus_opponent,p_raw_t,p_holm_t,p_raw_wilcoxon,p_holm_wilcoxon,rank_biserial_r\n']);
    for idx = 1:size(rows, 1)
        row = rows(idx, :);
        fprintf(fid, '%d,%s,%d,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n', ...
            row{1}, csvSafe(row{2}), row{3}, csvSafe(row{4}), row{5}, row{6}, row{7}, row{8}, row{9}, row{10}, row{11}, row{12});
    end

    clear cleanup;
end

function writeScenarioMarkdown(outputDir, scenarioIdx, scenarioDefinition, scenarioResults)
    summary = scenarioResults.summaryResults;
    pairwise = scenarioResults.pairwiseResults;
    filename = fullfile(outputDir, sprintf('scenario%d_summary.md', scenarioIdx));
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open markdown summary for writing: %s', filename);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, '# Scenario %d\n\n', scenarioIdx);
    fprintf(fid, '- Label: %s\n', scenarioDefinition.label);
    fprintf(fid, '- Terrain: %s\n', scenarioDefinition.terrainFile);
    fprintf(fid, '- Danger zones: %d\n', scenarioDefinition.numDangerZones);
    fprintf(fid, '- Metric: %s\n\n', scenarioResults.metricName);

    fprintf(fid, '## Means\n\n');
    fprintf(fid, '| Algorithm | Mean | Std | Median |\n');
    fprintf(fid, '|---|---:|---:|---:|\n');
    for idx = 1:numel(summary)
        row = summary(idx);
        fprintf(fid, '| %s | %.4f | %.4f | %.4f |\n', row.algorithm, row.mean, row.std, row.median);
    end

    fprintf(fid, '\n## Attractor-Field Pairwise\n\n');
    fprintf(fid, '| Opponent | Mean diff (RR - Opp) | p raw t | p raw Wilcoxon | p Holm Wilcoxon |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|\n');
    for idx = 1:numel(pairwise)
        row = pairwise(idx);
        if strcmp(row.algorithm_1, 'AFSACPSO')
            opponent = row.algorithm_2;
            meanDiff = row.mean_diff;
        elseif strcmp(row.algorithm_2, 'AFSACPSO')
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

function scenarioSet = normalizeScenarioSet(scenarios, dataDir)
    if isstruct(scenarios)
        scenarioSet = scenarios(:);
        return;
    end

    chrismasTerrainFile = fullfile(dataDir, 'ChrismasTerrain2.tif');
    scenarioSet = repmat(struct( ...
        'label', '', ...
        'terrainFile', chrismasTerrainFile, ...
        'numDangerZones', 0, ...
        'startPoint', [], ...
        'goalPoint', []), numel(scenarios), 1);

    for idx = 1:numel(scenarios)
        scenarioSet(idx).label = sprintf('ChrismasTerrain2 - Scenario %d', idx);
        scenarioSet(idx).numDangerZones = scenarios(idx);
    end
end

function summary = formatScenarioSummary(scenarios)
    labels = cell(numel(scenarios), 1);
    for idx = 1:numel(scenarios)
        labels{idx} = sprintf('%d:%s (%d DZ)', idx, scenarios(idx).label, scenarios(idx).numDangerZones);
    end
    summary = strjoin(labels, '; ');
end

function [scenarioDefinition, scenarioEnvironment] = resolveScenarioDefinition(scenarios, scenarioIdx, baseEnvironment)
    scenarioDefinition = scenarios(scenarioIdx);
    scenarioEnvironment = baseEnvironment;

    if isfield(scenarioDefinition, 'terrainFile') && ~isempty(scenarioDefinition.terrainFile)
        if shouldResetTerrainAnchors(baseEnvironment, scenarioDefinition.terrainFile)
            scenarioEnvironment = clearTerrainAnchors(scenarioEnvironment);
        end
        scenarioEnvironment.terrainFile = scenarioDefinition.terrainFile;
    end

    scenarioEnvironment = resolveTerrainEnvironment(scenarioEnvironment);
    scenarioEnvironment = applyPointOverrides(scenarioEnvironment, scenarioDefinition);

    if ~isfield(scenarioDefinition, 'label') || isempty(scenarioDefinition.label)
        scenarioDefinition.label = sprintf('Scenario %d', scenarioIdx);
    end
end

function tf = shouldResetTerrainAnchors(baseEnvironment, scenarioTerrainFile)
    tf = true;
    if isfield(baseEnvironment, 'terrainFile') && ~isempty(baseEnvironment.terrainFile)
        tf = ~strcmpi(char(string(baseEnvironment.terrainFile)), char(string(scenarioTerrainFile)));
    end
end

function environment = clearTerrainAnchors(environment)
    if isfield(environment, 'mapSize')
        environment.mapSize = [];
    end
    if isfield(environment, 'startPoint')
        environment.startPoint = [];
    end
    if isfield(environment, 'goalPoint')
        environment.goalPoint = [];
    end
end

function environment = applyPointOverrides(environment, scenarioDefinition)
    if isfield(scenarioDefinition, 'startPoint') && ~isempty(scenarioDefinition.startPoint)
        environment.startPoint = double(reshape(scenarioDefinition.startPoint, 1, 3));
    end
    if isfield(scenarioDefinition, 'goalPoint') && ~isempty(scenarioDefinition.goalPoint)
        environment.goalPoint = double(reshape(scenarioDefinition.goalPoint, 1, 3));
    end
end
