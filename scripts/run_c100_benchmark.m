function outputDir = run_c100_benchmark(numRuns, maxIterations, popSize, numWorkers, outputRoot)
%RUN_C100_BENCHMARK Benchmark table baselines + RRSACPSO on the c100 scenario.

    if nargin < 1 || isempty(numRuns)
        numRuns = 30;
    end
    if nargin < 2 || isempty(maxIterations)
        maxIterations = 1000;
    end
    if nargin < 3 || isempty(popSize)
        popSize = 100;
    end
    if nargin < 4 || isempty(numWorkers)
        numWorkers = 14;
    end

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    cd(repoRoot);

    addpath(repoRoot, '-begin');
    addpath(genpath(fullfile(repoRoot, 'algorithms')), '-begin');
    addpath(genpath(fullfile(repoRoot, 'scripts')), '-begin');
    addpath(genpath(fullfile(repoRoot, 'shared')), '-begin');
    addpath(fullfile(repoRoot, 'data'), '-begin');
    rehash path;

    if nargin < 5 || isempty(outputRoot)
        outputRoot = fullfile(repoRoot, 'benchmark');
    end
    ensureDirectory(outputRoot);

    stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    outputDir = fullfile(outputRoot, sprintf('c100_pop%d_iter%d_runs%d_workers%d_%s', ...
        popSize, maxIterations, numRuns, numWorkers, stamp));
    resultsDir = fullfile(outputDir, 'results');
    ensureDirectory(outputDir);
    ensureDirectory(resultsDir);

    logPath = fullfile(outputDir, 'benchmark.log');
    diary(logPath);
    cleanupDiary = onCleanup(@() diary('off'));

    fprintf('========================================\n');
    fprintf('  c100 Combined Benchmark Runner\n');
    fprintf('========================================\n');
    fprintf('Start time: %s\n', char(datetime('now')));
    fprintf('Output dir: %s\n', outputDir);
    fprintf('Results dir: %s\n', resultsDir);
    fprintf('Runs: %d | Population: %d | Iterations: %d | Workers: %d\n\n', ...
        numRuns, popSize, maxIterations, numWorkers);

    setenv('VIETANH_PARPOOL_WORKERS', num2str(numWorkers));
    setenv('VIETANH_RESULTS_DIR', resultsDir);
    setenv('APEXPSO_CONFIG_QUIET', '1');
    clear getResultsDir;

    initializeParallelPool('parallel');
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        error('Parallel pool was not created.');
    end
    fprintf('Parallel pool ready with %d workers.\n', poolobj.NumWorkers);

    scenario = resolveC100Scenario(repoRoot);
    algorithms = buildCombinedAlgorithms(popSize, maxIterations);
    terrainMap = defaultTerrainColormap();

    [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
        scenario.environment.mapSize, 1, scenario.definition.terrainFile);
    dangerZones = generateDangerZones( ...
        scenario.definition.numDangerZones, scenario.environment.mapSize, terrainGrid, terrainX, terrainY);

    commonBefore = { ...
        scenario.environment.startPoint, ...
        scenario.environment.goalPoint, ...
        dangerZones, ...
        terrainGrid, terrainX, terrainY, ...
        scenario.environment.mapSize ...
    };

    commonAfter = { ...
        scenario.environment.globalPlanInterval, ...
        scenario.environment.pathDeviationThreshold, ...
        scenario.environment.obstacleChangeThreshold, ...
        scenario.environment.timeStep, ...
        scenario.environment.totalTime, ...
        terrainMap ...
    };

    scenarioResults = cell(numRuns, 1);
    fprintf('Scheduling %d runs across %d algorithms on c100.\n', numRuns, numel(algorithms));
    parfor runIdx = 1:numRuns
        seedBase = runIdx * 1000 + 100;
        scenarioResults{runIdx} = runAlgorithmBatch(algorithms, seedBase, commonBefore, commonAfter);
    end

    rawResultsPath = fullfile(outputDir, 'all_run_results.mat');
    save(rawResultsPath, 'scenarioResults', 'algorithms', 'scenario', 'numRuns', 'maxIterations', 'popSize');

    performTTest(scenarioResults, algorithms, 1);
    generateReports(scenarioResults, algorithms, 1);

    manifest = struct();
    manifest.generatedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    manifest.outputDir = outputDir;
    manifest.resultsDir = resultsDir;
    manifest.logPath = logPath;
    manifest.rawResultsPath = rawResultsPath;
    manifest.numRuns = numRuns;
    manifest.maxIterations = maxIterations;
    manifest.popSize = popSize;
    manifest.numWorkers = numWorkers;
    manifest.scenarioLabel = scenario.definition.label;
    manifest.terrainFile = scenario.definition.terrainFile;
    manifest.goalPoint = scenario.environment.goalPoint;
    manifest.algorithmNames = string(cellfun(@(alg) alg.displayName, algorithms, 'UniformOutput', false));
    save(fullfile(outputDir, 'manifest.mat'), 'manifest');

    fprintf('\nBenchmark complete.\n');
    fprintf('Raw results: %s\n', rawResultsPath);
    fprintf('Manifest: %s\n', fullfile(outputDir, 'manifest.mat'));
    fprintf('Log: %s\n', logPath);
end

function scenario = resolveC100Scenario(repoRoot)
    scenarios = buildDefaultScenarioSet(fullfile(repoRoot, 'data'));
    matchIdx = find(contains({scenarios.label}, 'terrainStruct_c_100'), 1);
    if isempty(matchIdx)
        error('Could not locate terrainStruct_c_100 scenario.');
    end

    scenario = struct();
    scenario.definition = scenarios(matchIdx);
    scenario.environment = struct( ...
        'mapSize', [100, 100, 100], ...
        'startPoint', [10, 95, 10], ...
        'goalPoint', [97, 2, 10], ...
        'terrainFile', scenario.definition.terrainFile, ...
        'timeStep', 0.1, ...
        'totalTime', 0.1, ...
        'globalPlanInterval', 1e5, ...
        'pathDeviationThreshold', 1e5, ...
        'obstacleChangeThreshold', 1e5);
    scenario.environment = resolveTerrainEnvironment(scenario.environment);
    scenario.environment = applyScenarioPointOverrides(scenario.environment, scenario.definition);
end

function algorithms = buildCombinedAlgorithms(popSize, maxIterations)
    algorithms = {
        makeAlgorithm('RL-based', 'RLAMPSO (Global)', 'RLAMPSO_Online_Global', ...
            {'RLAMPSO', popSize, maxIterations, 0.7, 1.49, 1.49, '', 'baseline', 'global'})
        makeAlgorithm('RL-based', 'DQN-PSO (Global)', 'DQN_PSO_Online_Global', ...
            {'DQN_PSO', popSize, maxIterations, 0.7, 1.49, 1.49, '', 'global'})
        makeAlgorithm('RL-based', 'RRSACPSO', 'RRSACPSO_Online', ...
            {'RRSACPSO_Online', maxIterations})
        makeAlgorithm('RL-based', 'MPSORL (Multi-Strategy RL-PSO)', 'MPSORL', ...
            {'MPSORL', popSize, maxIterations, 0.9, 2.5, 2.5, 0.6, 0.8, 0.8, 50, 0.4})
        makeAlgorithm('RL-based', 'SAC-SAPSO', 'SACSAPSO_Paper', ...
            {'SACSAPSO_Paper', popSize, maxIterations, 15})
        makeAlgorithm('RL-based', 'PPO-PSO (Global)', 'PPO_PSO_Online_Global', ...
            {'PPO_PSO', popSize, maxIterations})
        makeAlgorithm('Others', 'PSO-Standard', 'PSO_Standard', ...
            {'PSO', popSize, maxIterations, 0.7, 1.49, 1.49})
        makeAlgorithm('Others', 'PSO-LDIW', 'PSO_LDIW', ...
            {'PSO_TVAC', popSize, maxIterations, 0.7, 1.49, 1.49})
        makeAlgorithm('Others', 'FIPS', 'FIPS', ...
            {'FIPS', popSize, maxIterations, 0.7, 1.49, 1.49})
        makeAlgorithm('Others', 'CLPSO', 'CLPSO', ...
            {'CLPSO', popSize, maxIterations, 0.7, 1.5})
        makeAlgorithm('Others', 'SAEPSO [11]', 'SAEPSO_Full', ...
            {'SAEPSO', popSize, maxIterations, 0.5, 2.5, 0.1, 0.9, true})
        makeAlgorithm('Others', 'SAEPSO* [11]', 'SAEPSO_ParamsOnly', ...
            {'SAEPSO', popSize, maxIterations, 0.5, 2.5, 0.1, 0.9, false})
        makeAlgorithm('Others', 'UAPSO [25]', 'UAPSO', ...
            {'UAPSO', popSize, maxIterations, 0.5, 2.5})
    };
    algorithms = finaliseAlgorithms(algorithms);
end

function runResults = runAlgorithmBatch(algorithms, seedBase, commonBefore, commonAfter)
    runResults = struct();
    for idx = 1:numel(algorithms)
        algorithm = algorithms{idx};
        rng(seedBase + algorithm.seedOffset);

        allParams = [commonBefore, algorithm.specificParams, commonAfter];
        [~, finalPath, pathLength, executionTime, ~, metrics] = evalc( ...
            'algorithm.function(allParams{:})');

        metrics.algorithmName = algorithm.displayName;
        metrics.pathLength = pathLength;
        metrics.executionTime = executionTime;

        runResults.(algorithm.fieldName) = metrics;
        runResults.([algorithm.fieldName '_finalPath']) = finalPath;
        runResults.([algorithm.fieldName '_globalPath']) = metrics.finalGlobalPath;
    end
end

function algorithm = makeAlgorithm(category, displayName, fieldName, specificParams)
    algorithm = struct();
    algorithm.category = category;
    algorithm.displayName = displayName;
    algorithm.fieldName = fieldName;
    algorithm.specificParams = specificParams;
    algorithm.function = @(varargin) directGlobalPlanning(varargin{:});
end

function algorithms = finaliseAlgorithms(algorithms)
    for idx = 1:numel(algorithms)
        algorithms{idx}.seedOffset = idx;
    end
end

function cmap = defaultTerrainColormap()
    cmap = [
        0.2 0.4 0.1;
        0.3 0.6 0.2;
        0.6 0.5 0.3;
        0.7 0.6 0.4;
        0.8 0.7 0.6;
        0.9 0.9 0.9
    ];
end

function ensureDirectory(directoryPath)
    if exist(directoryPath, 'dir') ~= 7
        mkdir(directoryPath);
    end
end
