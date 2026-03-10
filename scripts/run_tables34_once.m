clc;
clear;
close all;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
cd(repoRoot);

addpath(repoRoot, '-begin');
addpath(genpath(fullfile(repoRoot, 'algorithms')), '-begin');
addpath(genpath(fullfile(repoRoot, 'scripts')), '-begin');
addpath(genpath(fullfile(repoRoot, 'shared')), '-begin');
addpath(fullfile(repoRoot, 'data'), '-begin');
rehash path;

runStart = datetime('now');
fprintf('========================================\n');
fprintf('  Table 3 + Table 4 One-Shot Runner\n');
fprintf('========================================\n');
fprintf('Start time: %s\n', char(runStart));

targetWorkers = 14;
setenv('VIETANH_PARPOOL_WORKERS', num2str(targetWorkers));

outputRoot = fullfile(repoRoot, 'new result');
rlOutputDir = fullfile(outputRoot, 'rl_based');
othersOutputDir = fullfile(outputRoot, 'others_results');
ensureDirectory(outputRoot);
ensureDirectory(rlOutputDir);
ensureDirectory(othersOutputDir);

validateMatlabRuntime();
validateOutputDirectories({outputRoot, rlOutputDir, othersOutputDir});

initializeParallelPool('parallel');
poolobj = gcp('nocreate');
if isempty(poolobj)
    error('Parallel pool was not created.');
end
if poolobj.NumWorkers ~= targetWorkers
    error('Expected %d workers but got %d workers.', targetWorkers, poolobj.NumWorkers);
end

fprintf('Parallel pool ready with %d workers.\n', poolobj.NumWorkers);

environment = struct( ...
    'mapSize',                [100, 100, 100], ...
    'startPoint',             [10, 95, 10], ...
    'goalPoint',              [97, 2, 10], ...
    'terrainFile',            '', ...
    'timeStep',               0.1, ...
    'totalTime',              0.1, ...
    'globalPlanInterval',     1e5, ...
    'pathDeviationThreshold', 1e5, ...
    'obstacleChangeThreshold',1e5 ...
);
environment = resolveTerrainEnvironment(environment);

scenarios = [0; 5; 10];
numRuns = 30;

[algGroups, groupNames, groupOutputDirs] = buildTableGroups(rlOutputDir, othersOutputDir);
numGroups = numel(algGroups);
numScenarios = numel(scenarios);
numJobs = numGroups * numScenarios * numRuns;

[scenarioCommonBefore, scenarioCommonAfter] = buildScenarioCaches(environment, scenarios);
jobSpec = buildJobSpec(numGroups, numScenarios, numRuns);

algGroupsConst = parallel.pool.Constant(algGroups);
scenarioCommonBeforeConst = parallel.pool.Constant(scenarioCommonBefore);
scenarioCommonAfterConst = parallel.pool.Constant(scenarioCommonAfter);

jobTemplate = struct('groupIdx', 0, 'scenarioIdx', 0, 'runIdx', 0, 'runResults', struct());
jobOutputs = repmat(jobTemplate, numJobs, 1);

fprintf('Scheduling %d jobs (2 groups x 3 scenarios x %d runs) in one global queue.\n', numJobs, numRuns);

parfor jobIdx = 1:numJobs
    spec = jobSpec(jobIdx);
    groupIdx = spec.groupIdx;
    scenarioIdx = spec.scenarioIdx;
    runIdx = spec.runIdx;

    algorithms = algGroupsConst.Value{groupIdx};
    seedBase = runIdx * 1000 + scenarioIdx * 100;

    runResults = runAlgorithmBatch( ...
        algorithms, ...
        seedBase, ...
        scenarioCommonBeforeConst.Value{scenarioIdx}, ...
        scenarioCommonAfterConst.Value{scenarioIdx});

    out = jobTemplate;
    out.groupIdx = groupIdx;
    out.scenarioIdx = scenarioIdx;
    out.runIdx = runIdx;
    out.runResults = runResults;
    jobOutputs(jobIdx) = out;
end

fprintf('All jobs completed. Aggregating per-group/per-scenario outputs...\n');
aggregated = aggregateJobOutputs(jobOutputs, numGroups, numScenarios, numRuns);

for groupIdx = 1:numGroups
    setenv('VIETANH_RESULTS_DIR', groupOutputDirs{groupIdx});
    clear getResultsDir; % reset persistent cache

    algorithms = algGroups{groupIdx};
    fprintf('\n=== Exporting %s results to %s ===\n', groupNames{groupIdx}, groupOutputDirs{groupIdx});
    for scenarioIdx = 1:numScenarios
        scenarioResults = aggregated{groupIdx, scenarioIdx};
        performTTest(scenarioResults, algorithms, scenarioIdx);
        generateReports(scenarioResults, algorithms, scenarioIdx);
    end
end

setenv('VIETANH_RESULTS_DIR', '');
clear getResultsDir;

runEnd = datetime('now');
manifestPath = writeManifest(repoRoot, outputRoot, runStart, runEnd, groupNames, groupOutputDirs);

fprintf('\nRun complete.\n');
fprintf('End time: %s\n', char(runEnd));
fprintf('Manifest: %s\n', manifestPath);

function [algGroups, groupNames, groupOutputDirs] = buildTableGroups(rlOutputDir, othersOutputDir)
    defaults = defaultPsoParameters();

    rlBased = {
        makeAlgorithm('RL-based', 'RLAMPSO (Global)', 'RLAMPSO_Online_Global', {'RLAMPSO', defaults.popSize, defaults.maxIterations, 0.7, 1.49, 1.49, '', 'baseline', 'global'});
        makeAlgorithm('RL-based', 'DQN-PSO (Global)', 'DQN_PSO_Online_Global', {'DQN_PSO', defaults.popSize, defaults.maxIterations, 0.7, 1.49, 1.49, '', 'global'});
        makeAlgorithm('RL-based', 'RRSACPSO', 'RRSACPSO_Online', {'RRSACPSO_Online', defaults.maxIterations}, struct('requiresSerial', false));
        makeAlgorithm('RL-based', 'MPSORL (Multi-Strategy RL-PSO)', 'MPSORL', {'MPSORL', defaults.popSize, defaults.maxIterations, 0.9, 2.5, 2.5, 0.6, 0.8, 0.8, 50, 0.4});
        makeAlgorithm('RL-based', 'SAC-SAPSO', 'SACSAPSO_Paper', {'SACSAPSO_Paper', defaults.popSize, defaults.maxIterations, 15}, struct('requiresSerial', false));
        makeAlgorithm('RL-based', 'PPO-PSO (Global)', 'PPO_PSO_Online_Global', {'PPO_PSO', defaults.popSize, defaults.maxIterations}, struct('requiresSerial', false));
    };

    others = {
        makeAlgorithm('Others', 'RRSACPSO', 'RRSACPSO_Online', {'RRSACPSO_Online', defaults.maxIterations});
        makeAlgorithm('Others', 'PSO-Standard', 'PSO_Standard', {'PSO', defaults.popSize, defaults.maxIterations, 0.7, 1.49, 1.49});
        makeAlgorithm('Others', 'PSO-LDIW', 'PSO_LDIW', {'PSO_TVAC', defaults.popSize, defaults.maxIterations, 0.7, 1.49, 1.49});
        makeAlgorithm('Others', 'FIPS', 'FIPS', {'FIPS', defaults.popSize, defaults.maxIterations, 0.7, 1.49, 1.49});
        makeAlgorithm('Others', 'CLPSO', 'CLPSO', {'CLPSO', defaults.popSize, defaults.maxIterations, 0.7, 1.5});
        makeAlgorithm('Others', 'SAEPSO [11]', 'SAEPSO_Full', {'SAEPSO', defaults.popSize, defaults.maxIterations, 0.5, 2.5, 0.1, 0.9, true});
        makeAlgorithm('Others', 'SAEPSO* [11]', 'SAEPSO_ParamsOnly', {'SAEPSO', defaults.popSize, defaults.maxIterations, 0.5, 2.5, 0.1, 0.9, false});
        makeAlgorithm('Others', 'UAPSO [25]', 'UAPSO', {'UAPSO', defaults.popSize, defaults.maxIterations, 0.5, 2.5});
    };

    algGroups = {finaliseAlgorithms(rlBased), finaliseAlgorithms(others)};
    groupNames = {'RL-based (Table 3)', 'Others (Table 4)'};
    groupOutputDirs = {rlOutputDir, othersOutputDir};
end

function [scenarioCommonBefore, scenarioCommonAfter] = buildScenarioCaches(environment, scenarios)
    numScenarios = numel(scenarios);
    scenarioCommonBefore = cell(numScenarios, 1);
    scenarioCommonAfter = cell(numScenarios, 1);
    terrainMap = defaultTerrainColormap();

    for scenarioIdx = 1:numScenarios
        numDangerZones = scenarios(scenarioIdx);
        [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
            environment.mapSize, scenarioIdx, environment.terrainFile);
        dangerZones = generateDangerZones(numDangerZones, environment.mapSize, terrainGrid, terrainX, terrainY);

        scenarioCommonBefore{scenarioIdx} = { ...
            environment.startPoint, ...
            environment.goalPoint, ...
            dangerZones, ...
            terrainGrid, terrainX, terrainY, ...
            environment.mapSize ...
        };

        scenarioCommonAfter{scenarioIdx} = { ...
            environment.globalPlanInterval, ...
            environment.pathDeviationThreshold, ...
            environment.obstacleChangeThreshold, ...
            environment.timeStep, ...
            environment.totalTime, ...
            terrainMap ...
        };
    end
end

function jobSpec = buildJobSpec(numGroups, numScenarios, numRuns)
    total = numGroups * numScenarios * numRuns;
    template = struct('groupIdx', 0, 'scenarioIdx', 0, 'runIdx', 0);
    jobSpec = repmat(template, total, 1);

    idx = 0;
    for runIdx = 1:numRuns
        for groupIdx = 1:numGroups
            for scenarioIdx = 1:numScenarios
                idx = idx + 1;
                jobSpec(idx).groupIdx = groupIdx;
                jobSpec(idx).scenarioIdx = scenarioIdx;
                jobSpec(idx).runIdx = runIdx;
            end
        end
    end
end

function aggregated = aggregateJobOutputs(jobOutputs, numGroups, numScenarios, numRuns)
    aggregated = cell(numGroups, numScenarios);
    for groupIdx = 1:numGroups
        for scenarioIdx = 1:numScenarios
            aggregated{groupIdx, scenarioIdx} = cell(numRuns, 1);
        end
    end

    for idx = 1:numel(jobOutputs)
        row = jobOutputs(idx);
        aggregated{row.groupIdx, row.scenarioIdx}{row.runIdx} = row.runResults;
    end

    for groupIdx = 1:numGroups
        for scenarioIdx = 1:numScenarios
            for runIdx = 1:numRuns
                if isempty(aggregated{groupIdx, scenarioIdx}{runIdx})
                    error('Missing output for group %d scenario %d run %d.', groupIdx, scenarioIdx, runIdx);
                end
            end
        end
    end
end

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

function defaults = defaultPsoParameters()
    defaults = struct('popSize', 40, 'maxIterations', 1500);
end

function algorithm = makeAlgorithm(category, displayName, fieldName, specificParams, opts)
    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    algorithm = struct();
    algorithm.category = category;
    algorithm.displayName = displayName;
    algorithm.fieldName = fieldName;
    algorithm.specificParams = specificParams;
    algorithm.function = @(varargin) directGlobalPlanning(varargin{:});

    if isfield(opts, 'requiresSerial')
        algorithm.requiresSerial = logical(opts.requiresSerial);
    else
        algorithm.requiresSerial = false;
    end
end

function algorithms = finaliseAlgorithms(algorithms)
    for idx = 1:numel(algorithms)
        if ~isfield(algorithms{idx}, 'requiresSerial')
            algorithms{idx}.requiresSerial = false;
        end
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

function validateMatlabRuntime()
    runtimePath = matlabroot;
    if isempty(runtimePath) || exist(runtimePath, 'dir') ~= 7
        error('MATLAB runtime root is invalid: %s', runtimePath);
    end
    fprintf('MATLAB runtime: %s\n', runtimePath);
end

function validateOutputDirectories(paths)
    for i = 1:numel(paths)
        p = paths{i};
        if ~ensureWritableDirectory(p)
            error('Output directory is not writable: %s', p);
        end
    end
end

function ensureDirectory(directoryPath)
    if exist(directoryPath, 'dir') ~= 7
        mkdir(directoryPath);
    end
end

function ok = ensureWritableDirectory(directoryPath)
    ok = true;
    if exist(directoryPath, 'dir') ~= 7
        [ok, msg] = mkdir(directoryPath);
        if ~ok
            fprintf('Failed to create directory: %s\n', directoryPath);
            fprintf('Reason: %s\n', msg);
            return;
        end
    end

    testFile = fullfile(directoryPath, sprintf('.writetest_%s', char(datetime('now', 'Format', 'yyyyMMdd_HHmmssSSS'))));
    fid = fopen(testFile, 'w');
    if fid == -1
        ok = false;
        return;
    end
    fclose(fid);
    delete(testFile);
end

function manifestPath = writeManifest(repoRoot, outputRoot, runStart, runEnd, groupNames, groupOutputDirs)
    manifestPath = fullfile(outputRoot, sprintf('run_manifest_%s.txt', char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'))));
    fid = fopen(manifestPath, 'w');
    if fid == -1
        error('Could not write manifest file: %s', manifestPath);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'Table 3 + Table 4 one-shot run manifest\n');
    fprintf(fid, 'Start: %s\n', char(runStart));
    fprintf(fid, 'End: %s\n', char(runEnd));
    fprintf(fid, 'Duration: %s\n', char(runEnd - runStart));
    fprintf(fid, 'MATLAB: %s\n', version);
    fprintf(fid, 'Workers requested: 14\n');
    fprintf(fid, 'Command: matlab -batch "cd(''%s''); run(''scripts/run_tables34_once.m'');"\n', repoRoot);

    for idx = 1:numel(groupNames)
        fprintf(fid, '\n[%s]\n', groupNames{idx});
        fprintf(fid, 'OutputDir: %s\n', groupOutputDirs{idx});
        files = dir(fullfile(groupOutputDirs{idx}, 'TTest*'));
        if isempty(files)
            fprintf(fid, 'Files: (none)\n');
        else
            for f = 1:numel(files)
                fprintf(fid, '  %s\n', fullfile(files(f).folder, files(f).name));
            end
        end
    end

    clear cleanup;
end
