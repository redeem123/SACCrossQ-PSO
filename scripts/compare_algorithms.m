function compare_algorithms(config)
%COMPARE_ALGORITHMS Run the configured PSO variants across scenarios.
%
%   This refactored entry point accepts a single CONFIG struct produced by
%   LOAD_EXPERIMENT_CONFIG, keeping runtime parameters in one place and
%   slimming the public API for the comparison workflow.

    arguments
        config struct
    end

    algorithms   = config.algorithms;
    general      = config.general;
    environment  = config.environment;
    scenarios    = config.scenarios;
    executionMode = general.executionMode;
    numRuns       = general.numRuns;

    fprintf('Initialising parallel pool (%s mode)...\n', executionMode);
    initializeParallelPool(executionMode);

    numScenarios = numel(scenarios);
    allScenarioResults = cell(numScenarios, 1);

    for scenarioIdx = 1:numScenarios
        [scenarioDefinition, scenarioEnvironment] = resolveScenarioDefinition( ...
            scenarios, scenarioIdx, environment);
        numDangerZones = scenarioDefinition.numDangerZones;

        fprintf('\n==========================================================\n');
        fprintf('Scenario %d: %s\n', scenarioIdx, scenarioDefinition.label);
        fprintf('Terrain: %s\n', scenarioDefinition.terrainFile);
        fprintf('Danger Zones: %d\n', numDangerZones);
        fprintf('==========================================================\n');

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
            scenarioEnvironment.mapSize, scenarioIdx, scenarioEnvironment.terrainFile);
        dangerZones = generateDangerZones(numDangerZones, scenarioEnvironment.mapSize, terrainGrid, terrainX, terrainY);

        terrain_map = defaultTerrainColormap();

        commonBefore = { ...
            scenarioEnvironment.startPoint, ...
            scenarioEnvironment.goalPoint, ...
            dangerZones, ...
            terrainGrid, terrainX, terrainY, ...
            scenarioEnvironment.mapSize ...
        };

        commonAfter = { ...
            scenarioEnvironment.globalPlanInterval, ...
            scenarioEnvironment.pathDeviationThreshold, ...
            scenarioEnvironment.obstacleChangeThreshold, ...
            scenarioEnvironment.timeStep, ...
            scenarioEnvironment.totalTime, ...
            terrain_map ...
        };

        scenarioResults = executeScenario( ...
            algorithms, executionMode, numRuns, scenarioIdx, ...
            commonBefore, commonAfter);

        allScenarioResults{scenarioIdx} = scenarioResults;

        fprintf('All runs completed. Performing statistical analysis...\n');
        performTTest(scenarioResults, algorithms, scenarioIdx);

        bestRunIdx = findBestRun(scenarioResults, algorithms);
        bestResults = scenarioResults{bestRunIdx};

        disp('Generating diagnostic visualisations...');
        debugPSOVariants(bestResults, algorithms);
        ComparePSOVariants(bestResults, algorithms);

        disp('Generating trajectory and convergence figures...');
        createUAVTrajectoryAnimation(bestResults, algorithms, dangerZones, terrainGrid, terrainX, terrainY, ...
            scenarioEnvironment.startPoint, scenarioEnvironment.goalPoint, scenarioEnvironment.mapSize, terrain_map, scenarioIdx);
        generateConvergencePlot(bestResults, algorithms, scenarioIdx);
        generateParameterTrackingPlot(bestResults, algorithms, scenarioIdx);
        generateRewardPlot(bestResults, algorithms, scenarioIdx);
        generateCriticLossPlot(bestResults, algorithms, scenarioIdx);

        generateReports(scenarioResults, algorithms, scenarioIdx);

        fprintf('Scenario %d complete.\n', scenarioIdx);
    end

    disp('All scenarios completed successfully!');
end

% -------------------------------------------------------------------------
function scenarioResults = executeScenario(algorithms, executionMode, numRuns, scenarioIdx, commonBefore, commonAfter)
    scenarioResults = cell(numRuns, 1);
    seedScenarioOffset = scenarioIdx * 100;

    if strcmpi(executionMode, 'serial')
        fprintf('Executing %d independent runs in SERIAL mode...\n', numRuns);
        for run = 1:numRuns
            seedBase = run * 1000 + seedScenarioOffset;
            scenarioResults{run} = runAlgorithmBatch(algorithms, seedBase, commonBefore, commonAfter, scenarioIdx, run);
            fprintf('  Completed Run %d/%d\n', run, numRuns);
        end
        return;
    end

    [parallelAlgorithms, serialAlgorithms] = splitAlgorithms(algorithms);

    if ~isempty(parallelAlgorithms)
        fprintf('Executing %d runs for %d parallel-safe algorithms...\n', numRuns, numel(parallelAlgorithms));
        parallelResults = cell(numRuns, 1);
        parfor run = 1:numRuns
            seedBase = run * 1000 + seedScenarioOffset;
            parallelResults{run} = runAlgorithmBatch(parallelAlgorithms, seedBase, commonBefore, commonAfter, scenarioIdx, run);
        end
        scenarioResults = parallelResults;
    else
        % Ensure the cell array is initialised for later merging.
        scenarioResults = cell(numRuns, 1);
    end

    if ~isempty(serialAlgorithms)
        fprintf('Executing %d runs for %d serial-only algorithms...\n', numRuns, numel(serialAlgorithms));
        for run = 1:numRuns
            seedBase = run * 1000 + seedScenarioOffset;
            serialResults = runAlgorithmBatch(serialAlgorithms, seedBase, commonBefore, commonAfter, scenarioIdx, run);
            scenarioResults{run} = mergeResults(scenarioResults{run}, serialResults);
        end
    end
end

% -------------------------------------------------------------------------
function runResults = runAlgorithmBatch(algorithms, seedBase, commonBefore, commonAfter, scenarioIdx, runIdx)
    checkpointPath = getRunCheckpointPath(scenarioIdx, runIdx);
    runResults = loadRunCheckpoint(checkpointPath);

    for idx = 1:numel(algorithms)
        algorithm = algorithms{idx};
        if isfield(runResults, algorithm.fieldName) && isfield(runResults, [algorithm.fieldName '_finalPath']) ...
                && isfield(runResults, [algorithm.fieldName '_globalPath'])
            fprintf('  Resuming run %d scenario %d: skipping completed %s\n', ...
                runIdx, scenarioIdx, algorithm.displayName);
            continue;
        end

        rng(seedBase + algorithm.seedOffset);

        allParams = [commonBefore, algorithm.specificParams, commonAfter];
        [finalPath, pathLength, executionTime, ~, metrics] = algorithm.function(allParams{:});

        metrics.algorithmName = algorithm.displayName;
        metrics.pathLength = pathLength;
        metrics.executionTime = executionTime;

        runResults.(algorithm.fieldName) = metrics;
        runResults.([algorithm.fieldName '_finalPath']) = finalPath;
        runResults.([algorithm.fieldName '_globalPath']) = metrics.finalGlobalPath;
        saveRunCheckpoint(checkpointPath, runResults);
    end
end

% -------------------------------------------------------------------------
function merged = mergeResults(primary, secondary)
    if isempty(primary)
        merged = secondary;
        return;
    end

    merged = primary;
    fields = fieldnames(secondary);
    for idx = 1:numel(fields)
        merged.(fields{idx}) = secondary.(fields{idx});
    end
end

% -------------------------------------------------------------------------
function [parallelAlgorithms, serialAlgorithms] = splitAlgorithms(algorithms)
    parallelAlgorithms = {};
    serialAlgorithms = {};

    for idx = 1:numel(algorithms)
        if algorithms{idx}.requiresSerial
            serialAlgorithms{end + 1} = algorithms{idx}; %#ok<AGROW>
        else
            parallelAlgorithms{end + 1} = algorithms{idx}; %#ok<AGROW>
        end
    end
end

% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
function checkpointPath = getRunCheckpointPath(scenarioIdx, runIdx)
    checkpointDir = fullfile(getResultsDir(), 'resume_checkpoints');
    if exist(checkpointDir, 'dir') ~= 7
        mkdir(checkpointDir);
    end
    checkpointPath = fullfile(checkpointDir, sprintf('scenario%d_run%d.mat', scenarioIdx, runIdx));
end

% -------------------------------------------------------------------------
function runResults = loadRunCheckpoint(checkpointPath)
    runResults = struct();
    if exist(checkpointPath, 'file') ~= 2
        return;
    end

    loaded = load(checkpointPath, 'runResults');
    if isfield(loaded, 'runResults') && isstruct(loaded.runResults)
        runResults = loaded.runResults;
    end
end

% -------------------------------------------------------------------------
function saveRunCheckpoint(checkpointPath, runResults)
    save(checkpointPath, 'runResults');
end

% -------------------------------------------------------------------------
function [scenarioDefinition, scenarioEnvironment] = resolveScenarioDefinition(scenarios, scenarioIdx, baseEnvironment)
    if isstruct(scenarios)
        scenarioDefinition = scenarios(scenarioIdx);
    else
        scenarioDefinition = struct('numDangerZones', scenarios(scenarioIdx));
    end

    if ~isfield(scenarioDefinition, 'numDangerZones') || isempty(scenarioDefinition.numDangerZones)
        error('Scenario %d is missing numDangerZones.', scenarioIdx);
    end

    scenarioEnvironment = baseEnvironment;
    if isfield(scenarioDefinition, 'terrainFile') && ~isempty(scenarioDefinition.terrainFile)
        if shouldResetTerrainAnchors(baseEnvironment, scenarioDefinition.terrainFile)
            scenarioEnvironment = clearTerrainAnchors(scenarioEnvironment);
        end
        scenarioEnvironment.terrainFile = scenarioDefinition.terrainFile;
    end
    scenarioEnvironment = resolveTerrainEnvironment(scenarioEnvironment);
    scenarioEnvironment = applyScenarioPointOverrides(scenarioEnvironment, scenarioDefinition);

    if ~isfield(scenarioDefinition, 'terrainFile') || isempty(scenarioDefinition.terrainFile)
        scenarioDefinition.terrainFile = scenarioEnvironment.terrainFile;
    end
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
