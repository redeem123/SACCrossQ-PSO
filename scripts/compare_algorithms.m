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

    numScenarios = size(scenarios, 1);
    allScenarioResults = cell(numScenarios, 1);

    for scenarioIdx = 1:numScenarios
        % Get number of danger zones for this scenario
        numDangerZones = scenarios(scenarioIdx);

        fprintf('\n==========================================================\n');
        fprintf('Scenario %d: %d Danger Zones\n', scenarioIdx, numDangerZones);
        fprintf('==========================================================\n');

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain(environment.mapSize, scenarioIdx);
        dangerZones = generateDangerZones(numDangerZones, environment.mapSize, terrainGrid, terrainX, terrainY);

        terrain_map = defaultTerrainColormap();

        commonBefore = { ...
            environment.startPoint, ...
            environment.goalPoint, ...
            dangerZones, ...
            terrainGrid, terrainX, terrainY, ...
            environment.mapSize ...
        };

        commonAfter = { ...
            environment.globalPlanInterval, ...
            environment.pathDeviationThreshold, ...
            environment.obstacleChangeThreshold, ...
            environment.timeStep, ...
            environment.totalTime, ...
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
            environment.startPoint, environment.goalPoint, environment.mapSize, terrain_map, scenarioIdx);
        generateConvergencePlot(bestResults, algorithms, scenarioIdx);
        generateParameterTrackingPlot(bestResults, algorithms, scenarioIdx);

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
            scenarioResults{run} = runAlgorithmBatch(algorithms, seedBase, commonBefore, commonAfter);
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
            parallelResults{run} = runAlgorithmBatch(parallelAlgorithms, seedBase, commonBefore, commonAfter);
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
            serialResults = runAlgorithmBatch(serialAlgorithms, seedBase, commonBefore, commonAfter);
            scenarioResults{run} = mergeResults(scenarioResults{run}, serialResults);
        end
    end
end

% -------------------------------------------------------------------------
function runResults = runAlgorithmBatch(algorithms, seedBase, commonBefore, commonAfter)
    runResults = struct();
    for idx = 1:numel(algorithms)
        algorithm = algorithms{idx};
        rng(seedBase + algorithm.seedOffset);

        allParams = [commonBefore, algorithm.specificParams, commonAfter];
        [finalPath, pathLength, executionTime, ~, metrics] = algorithm.function(allParams{:});

        metrics.algorithmName = algorithm.displayName;
        metrics.pathLength = pathLength;
        metrics.executionTime = executionTime;

        runResults.(algorithm.fieldName) = metrics;
        runResults.([algorithm.fieldName '_finalPath']) = finalPath;
        runResults.([algorithm.fieldName '_globalPath']) = metrics.finalGlobalPath;
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

