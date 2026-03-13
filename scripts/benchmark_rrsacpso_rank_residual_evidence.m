function benchmark_rrsacpso_rank_residual_evidence(numRuns, maxIterations, popSize, scenarios, numWorkers)
%BENCHMARK_RRSACPSO_RANK_RESIDUAL_EVIDENCE
% Run the missing rank-residual context and scale ablations and export
% table-ready rows for paper/main.tex.

    if nargin < 1 || isempty(numRuns)
        numRuns = 30;
    end
    if nargin < 2 || isempty(maxIterations)
        maxIterations = 1000;
    end
    if nargin < 3 || isempty(popSize)
        popSize = 100;
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
        sprintf('rrsacpso_rank_residual_evidence_%s', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end

    oldQuiet = getenv('APEXPSO_CONFIG_QUIET');
    quietCleanup = onCleanup(@() setenv('APEXPSO_CONFIG_QUIET', oldQuiet));
    setenv('APEXPSO_CONFIG_QUIET', '1');

    oldResultsDir = getenv('VIETANH_RESULTS_DIR');
    resultsCleanup = onCleanup(@() setenv('VIETANH_RESULTS_DIR', oldResultsDir));

    baseEnvironment = defaultEnvironment(dataDir);
    scenarioSet = normalizeScenarioSet(scenarios, dataDir);
    baseOverrides = buildBaseOverrides(popSize);
    families = buildFamilies(baseOverrides);

    fprintf('============================================================\n');
    fprintf('RRSACPSO Rank-Residual Evidence Benchmark\n');
    fprintf('============================================================\n');
    fprintf('Runs: %d\n', numRuns);
    fprintf('Max iterations: %d\n', maxIterations);
    fprintf('Population size: %d\n', popSize);
    fprintf('Workers: %d\n', max(1, numWorkers));
    fprintf('Scenarios: %s\n', formatScenarioSummary(scenarioSet));
    fprintf('Results: %s\n\n', outputDir);

    combinedRows = repmat(emptyCombinedRow(), 0, 1);

    for familyIdx = 1:numel(families)
        family = families(familyIdx);
        familyDir = fullfile(outputDir, family.id);
        familyResultsDir = fullfile(familyDir, 'results');
        ensureDir(familyDir);
        ensureDir(familyResultsDir);
        fprintf('\n============================================================\n');
        fprintf('Family: %s\n', family.name);
        fprintf('============================================================\n');

        for scenarioIdx = 1:numel(scenarioSet)
            [scenarioDefinition, scenarioEnvironment] = resolveScenarioDefinition( ...
                scenarioSet, scenarioIdx, baseEnvironment);
            numDangerZones = scenarioDefinition.numDangerZones;
            fprintf('\nScenario %d: %s\n', scenarioIdx, scenarioDefinition.label);
            fprintf('Danger zones: %d\n', numDangerZones);
            fprintf('Terrain: %s\n', scenarioDefinition.terrainFile);

            [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
                scenarioEnvironment.mapSize, scenarioIdx, scenarioEnvironment.terrainFile);
            dangerZones = generateDangerZones(numDangerZones, scenarioEnvironment.mapSize, terrainGrid, terrainX, terrainY);

            runResults = executeScenarioRuns(numRuns, family.variants, scenarioEnvironment, ...
                dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides, scenarioIdx, useParallel);

            setenv('VIETANH_RESULTS_DIR', familyResultsDir);
            performTTest(runResults, family.variants, scenarioIdx);
            scenarioResults = load(fullfile(familyResultsDir, sprintf('TTest_Results_Scenario%d.mat', scenarioIdx)));

            save(fullfile(familyDir, sprintf('%s_scenario%d_run_results.mat', family.id, scenarioIdx)), ...
                'runResults', 'scenarioIdx', 'scenarioDefinition', 'scenarioEnvironment', ...
                'numDangerZones', 'numRuns', 'maxIterations', 'popSize');

            timeSummary = summarizeExecutionTimes(runResults, family.variants);
            familyRows = buildCombinedRowsForScenario(family, scenarioIdx, scenarioResults, timeSummary);
            combinedRows = [combinedRows; familyRows]; %#ok<AGROW>
        end
    end

    writeCombinedArtifacts(outputDir, combinedRows, numel(scenarioSet));
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
    overrides.residualActionScale = struct('w', 0.18, 'c1', 0.55, 'c2', 0.55);
end

function families = buildFamilies(baseOverrides)
    baseScale = baseOverrides.residualActionScale;

    contextVariants = {
        makeVariant('RRSACPSO', 'RRSACPSO', 'rank-residual', struct(), '')
        makeVariant('RRSACPSO (No Rank r)', 'RRSACPSO_NoRank', 'rank-residual', ...
            struct('rankResidualContextMask', struct('rank', false, 'velocity', true, 'stagnation', true)), ...
            'RRSACPSO (No Rank $r$)')
        makeVariant('RRSACPSO (No Velocity u)', 'RRSACPSO_NoVelocity', 'rank-residual', ...
            struct('rankResidualContextMask', struct('rank', true, 'velocity', false, 'stagnation', true)), ...
            'RRSACPSO (No Velocity $u$)')
        makeVariant('RRSACPSO (No Stagnation s)', 'RRSACPSO_NoStagnation', 'rank-residual', ...
            struct('rankResidualContextMask', struct('rank', true, 'velocity', true, 'stagnation', false)), ...
            'RRSACPSO (No Stagnation $s$)')
    };

    scaleVariants = {
        makeVariant('RRSACPSO', 'RRSACPSO', 'rank-residual', struct(), '')
        makeVariant('0.5x Bounds Scaling', 'BoundsScale_0p5x', 'rank-residual', ...
            struct('residualActionScale', scaleStruct(baseScale, 0.5)), ...
            '$0.5\times$ Bounds Scaling')
        makeVariant('1.5x Bounds Scaling', 'BoundsScale_1p5x', 'rank-residual', ...
            struct('residualActionScale', scaleStruct(baseScale, 1.5)), ...
            '$1.5\times$ Bounds Scaling')
    };

    families = [ ...
        struct('id', 'context_ablation', 'name', 'Context Ablation', 'variants', {contextVariants}); ...
        struct('id', 'residual_scale_sensitivity', 'name', 'Residual Scale Sensitivity', 'variants', {scaleVariants}) ...
    ];
end

function scaled = scaleStruct(baseScale, factor)
    scaled = struct( ...
        'w', baseScale.w * factor, ...
        'c1', baseScale.c1 * factor, ...
        'c2', baseScale.c2 * factor);
end

function variant = makeVariant(displayName, fieldName, paramMode, overrides, paperRowLabel)
    variant = struct();
    variant.displayName = displayName;
    variant.fieldName = fieldName;
    variant.paramMode = paramMode;
    variant.overrides = overrides;
    variant.paperRowLabel = paperRowLabel;
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

function runResults = runVariantBatch(variants, seedBase, environment, dangerZones, terrainGrid, terrainX, terrainY, maxIterations, baseOverrides)
    runResults = struct();
    startPoint = environment.startPoint;
    goalPoint = environment.goalPoint;
    mapSize = environment.mapSize;
    for idx = 1:numel(variants)
        variant = variants{idx};
        rng(seedBase, 'twister');
        overrides = mergeStruct(baseOverrides, variant.overrides);
        params = struct( ...
            'maxIterations', maxIterations, ...
            'paramMode', variant.paramMode, ...
            'configOverrides', overrides);

        stepTimer = tic;
        [~, globalPath, algorithmSpecificStats] = evalc( ...
            'callApexVariant(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params)');
        executionTime = toc(stepTimer);

        metrics = buildMetrics(globalPath, algorithmSpecificStats, executionTime, variant.displayName);
        runResults.(variant.fieldName) = metrics;
    end
end

function [globalPath, algorithmSpecificStats] = callApexVariant(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params)
    [globalPath, algorithmSpecificStats] = callGlobalPlanningAlgorithm( ...
        'RRSACPSO_Online', startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params);
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

function timeSummary = summarizeExecutionTimes(runResults, variants)
    timeSummary = repmat(struct('algorithm', '', 'meanTime', NaN, 'stdTime', NaN), numel(variants), 1);
    for idx = 1:numel(variants)
        fieldName = variants{idx}.fieldName;
        values = NaN(numel(runResults), 1);
        for runIdx = 1:numel(runResults)
            if isfield(runResults{runIdx}, fieldName) && isfield(runResults{runIdx}.(fieldName), 'executionTime')
                values(runIdx) = runResults{runIdx}.(fieldName).executionTime;
            end
        end
        validValues = values(isfinite(values));
        timeSummary(idx).algorithm = variants{idx}.displayName;
        timeSummary(idx).meanTime = safeMean(validValues);
        timeSummary(idx).stdTime = safeStd(validValues);
    end
end

function rows = buildCombinedRowsForScenario(family, scenarioIdx, scenarioResults, timeSummary)
    rows = repmat(emptyCombinedRow(), 0, 1);
    baselineName = 'RRSACPSO';
    for idx = 1:numel(family.variants)
        variant = family.variants{idx};
        if isempty(variant.paperRowLabel)
            continue;
        end

        [meanFit, stdFit] = lookupSummaryStats(scenarioResults.summaryResults, variant.displayName);
        meanTime = lookupTimeSummary(timeSummary, variant.displayName);
        ttestMarker = lookupPairwiseMarker(scenarioResults.summaryResults, scenarioResults.pairwiseResults, baselineName, variant.displayName);

        row = emptyCombinedRow();
        row.family_id = family.id;
        row.row_label = variant.paperRowLabel;
        row.algorithm_name = variant.displayName;
        row.scenario_id = scenarioIdx;
        row.mean_fitness = meanFit;
        row.std_fitness = stdFit;
        row.mean_time = meanTime;
        row.ttest_marker = ttestMarker;
        rows(end + 1, 1) = row; %#ok<AGROW>
    end
end

function [meanFit, stdFit] = lookupSummaryStats(summaryResults, algorithmName)
    idx = find(strcmp({summaryResults.algorithm}, algorithmName), 1, 'first');
    if isempty(idx)
        error('Missing summary stats for algorithm "%s".', algorithmName);
    end
    meanFit = summaryResults(idx).mean;
    stdFit = summaryResults(idx).std;
end

function meanTime = lookupTimeSummary(timeSummary, algorithmName)
    idx = find(strcmp({timeSummary.algorithm}, algorithmName), 1, 'first');
    if isempty(idx)
        error('Missing time summary for algorithm "%s".', algorithmName);
    end
    meanTime = timeSummary(idx).meanTime;
end

function marker = lookupPairwiseMarker(summaryResults, pairwiseResults, baselineName, algorithmName)
    if strcmp(baselineName, algorithmName)
        marker = '---';
        return;
    end

    baseIdx = find(strcmp({summaryResults.algorithm}, baselineName), 1, 'first');
    algIdx = find(strcmp({summaryResults.algorithm}, algorithmName), 1, 'first');
    if isempty(baseIdx) || isempty(algIdx)
        error('Missing baseline or algorithm summary for marker lookup.');
    end

    baselineMean = summaryResults(baseIdx).mean;
    algorithmMean = summaryResults(algIdx).mean;
    pairIdx = find( ...
        (strcmp({pairwiseResults.algorithm_1}, baselineName) & strcmp({pairwiseResults.algorithm_2}, algorithmName)) | ...
        (strcmp({pairwiseResults.algorithm_1}, algorithmName) & strcmp({pairwiseResults.algorithm_2}, baselineName)), ...
        1, 'first');
    if isempty(pairIdx)
        error('Missing pairwise result for "%s" vs "%s".', baselineName, algorithmName);
    end

    isSignificant = pairwiseResults(pairIdx).p_holm_t < 0.05;
    if algorithmMean > baselineMean
        if isSignificant
            marker = '$\uparrow\uparrow$';
        else
            marker = '$\uparrow$';
        end
    elseif algorithmMean < baselineMean
        if isSignificant
            marker = '$\downarrow\downarrow$';
        else
            marker = '$\downarrow$';
        end
    else
        marker = '---';
    end
end

function writeCombinedArtifacts(outputDir, combinedRows, numScenarios)
    csvPath = fullfile(outputDir, 'paper_rank_residual_missing_rows.csv');
    texPath = fullfile(outputDir, 'paper_rank_residual_missing_rows.tex');
    writeCombinedCsv(csvPath, combinedRows);
    writeCombinedTex(texPath, combinedRows, numScenarios);
end

function writeCombinedCsv(csvPath, combinedRows)
    fid = fopen(csvPath, 'w');
    if fid == -1
        error('Could not open CSV for writing: %s', csvPath);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'family_id,row_label,algorithm_name,scenario_id,mean_fitness,std_fitness,mean_time,ttest_marker\n');
    for idx = 1:numel(combinedRows)
        row = combinedRows(idx);
        fprintf(fid, '%s,%s,%s,%d,%.10g,%.10g,%.10g,%s\n', ...
            csvSafe(row.family_id), ...
            csvSafe(row.row_label), ...
            csvSafe(row.algorithm_name), ...
            row.scenario_id, ...
            row.mean_fitness, ...
            row.std_fitness, ...
            row.mean_time, ...
            csvSafe(row.ttest_marker));
    end

    clear cleanup;
end

function writeCombinedTex(texPath, combinedRows, numScenarios)
    rowOrder = { ...
        'RRSACPSO (No Rank $r$)', ...
        'RRSACPSO (No Velocity $u$)', ...
        'RRSACPSO (No Stagnation $s$)', ...
        '$0.5\times$ Bounds Scaling', ...
        '$1.5\times$ Bounds Scaling' ...
    };

    fid = fopen(texPath, 'w');
    if fid == -1
        error('Could not open TeX rows file for writing: %s', texPath);
    end
    cleanup = onCleanup(@() fclose(fid));

    for rowIdx = 1:numel(rowOrder)
        rowLabel = rowOrder{rowIdx};
        rowLines = cell(1, 1 + 3 * numScenarios);
        rowLines{1} = rowLabel;
        for scenarioIdx = 1:numScenarios
            matchIdx = find(strcmp({combinedRows.row_label}, rowLabel) & [combinedRows.scenario_id] == scenarioIdx, 1, 'first');
            if isempty(matchIdx)
                error('Missing combined row for "%s" scenario %d.', rowLabel, scenarioIdx);
            end
            row = combinedRows(matchIdx);
            base = 2 + (scenarioIdx - 1) * 3;
            rowLines{base} = sprintf('%.2f $\\pm$ %.2f', row.mean_fitness, row.std_fitness);
            rowLines{base + 1} = sprintf('%.2f', row.mean_time);
            rowLines{base + 2} = row.ttest_marker;
        end
        fprintf(fid, '%s \\\\\n', strjoin(rowLines, ' & '));
    end

    clear cleanup;
end

function merged = mergeStruct(baseStruct, overrideStruct)
    merged = baseStruct;
    if nargin < 2 || isempty(overrideStruct)
        return;
    end

    fields = fieldnames(overrideStruct);
    for idx = 1:numel(fields)
        fieldName = fields{idx};
        merged.(fieldName) = overrideStruct.(fieldName);
    end
end

function row = emptyCombinedRow()
    row = struct( ...
        'family_id', '', ...
        'row_label', '', ...
        'algorithm_name', '', ...
        'scenario_id', NaN, ...
        'mean_fitness', NaN, ...
        'std_fitness', NaN, ...
        'mean_time', NaN, ...
        'ttest_marker', '');
end

function ensureDir(pathIn)
    if exist(pathIn, 'dir') ~= 7
        mkdir(pathIn);
    end
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

function textOut = csvSafe(textIn)
    if isempty(textIn)
        textOut = '';
        return;
    end
    textOut = strrep(char(string(textIn)), ',', ' ');
end

function value = safeMean(values)
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function value = safeStd(values)
    if numel(values) <= 1
        value = 0;
    else
        value = std(values);
    end
end
