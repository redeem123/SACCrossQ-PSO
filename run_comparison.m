clc;
clear;
close all;

repoRoot = fileparts(mfilename('fullpath'));

addpath(repoRoot);
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'algorithms')));
addpath(genpath(fullfile(repoRoot, 'scripts')));
addpath(genpath(fullfile(repoRoot, 'shared')));
addpath(fullfile(repoRoot, 'data'));

config = load_experiment_config(repoRoot);

ensureDirectory(config.paths.outputs);
ensureDirectory(config.paths.results);
ensureDirectory(config.paths.logs);

logFile = startLogging(config.paths.logs);

fprintf('========================================\n');
fprintf('  PSO Algorithm Comparison Launcher\n');
fprintf('========================================\n\n');
fprintf('Execution mode: %s\n', config.general.executionMode);
fprintf('Independent runs: %d\n', config.general.numRuns);
fprintf('Logging to: %s\n\n', logFile);

try
    compare_algorithms(config);
    fprintf('\nComparison complete! Log file saved to: %s\n', logFile);
catch ME
    reportFailure(ME, logFile);
    stopDiary();
    rethrow(ME);
end
stopDiary();

function config = load_experiment_config(repoRoot)

    if nargin < 1 || isempty(repoRoot)
        repoRoot = fileparts(fileparts(mfilename('fullpath')));
    end

    config.general = struct( ...
        'executionMode', getenvStringOrDefault('VIETANH_EXECUTION_MODE', 'parallel'), ...   % Options: 'parallel' | 'serial'
        'numRuns',        getenvOrDefault('VIETANH_NUM_RUNS', 30) ...          % Independent runs for statistics
    );

    outputsRoot = fullfile(repoRoot, 'outputs');

    config.paths = struct( ...
        'repoRoot',  repoRoot, ...
        'outputs',   outputsRoot, ...
        'results',   fullfile(outputsRoot, 'results'), ...
        'logs',      fullfile(outputsRoot, 'logs'), ...
        'data',      fullfile(repoRoot, 'data'), ...
        'models',    fullfile(repoRoot, 'models') ...
    );

    % NOTE: Using directGlobalPlanning - these parameters are not used
    % Kept for compatibility with existing code structure
    config.environment = struct( ...
        'mapSize',                [100, 100, 100], ...
        'startPoint',             [10, 95, 10], ...
        'goalPoint',              [97, 2, 10], ...
        'terrainFile',            '', ...
        'timeStep',               0.1, ...       % Not used in direct planning
        'totalTime',              0.1, ...       % Not used in direct planning
        'globalPlanInterval',     1e5, ...       % Not used in direct planning
        'pathDeviationThreshold', 1e5, ...       % Not used in direct planning
        'obstacleChangeThreshold',1e5 ...        % Not used in direct planning
    );
    config.environment = resolveTerrainEnvironment(config.environment);

    config.scenarios = buildDefaultScenarioSet(config.paths.data);
    config.scenarios = filterScenarioSet(config.scenarios);

    config.algorithms = buildAlgorithmRegistry(config.paths);
end

% -------------------------------------------------------------------------
function algorithms = buildAlgorithmRegistry(paths)
    defaults = defaultPsoParameters(paths);

    rlBased = {
        makeAlgorithm('RL-based', ...
            'RLAMPSO [36]', ...
            'RLAMPSO_Online_Global', ...
            {'RLAMPSO', defaults.popSize, defaults.maxIterations, defaults.rlam.initialW, defaults.rlam.initialC1, defaults.rlam.initialC2, '', defaults.rlam.configMode, 'global'});

        makeAlgorithm('RL-based', ...
            'DQN-PSO [34]', ...
            'DQN_PSO_Online_Global', ...
            {'DQN_PSO', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2, '', 'global'});

        makeAlgorithm('RL-based', ...
            'SAC-SAPSO [40]', ...
            'SACSAPSO_Paper', ...
            {'SACSAPSO_Paper', 30, defaults.maxIterations, 125}, ...
            struct('requiresSerial', true));
        makeAlgorithm('RL-based', ...
            'PPO-PSO', ...
            'PPO_PSO_Online_Global', ...
            {'PPO_PSO', defaults.popSize, defaults.maxIterations});
        makeAlgorithm('RL-based', ...
            'MPSORL', ...
            'MPSORL', ...
            {'MPSORL', defaults.popSize, defaults.maxIterations, 0.9, 2.5, 2.5, 0.6, 0.8, 0.8, 50, 0.4});
        makeAlgorithm('RL-based', ...
            'RRSACPSO', ...
            'RRSACPSO_Online', ...
            {'RRSACPSO_Online', defaults.maxIterations}, ...
            struct('requiresSerial', true));
    };

    ablationTrain = {};

    trainOtherDRL = {
        % %--------------RLAMPSO Training (saves to models/RLAMPSO/)-----------------------%
        % makeAlgorithm('Train-OtherDRL', ...
        %     'RLAMPSO Training (Global)', ...
        %     'RLAMPSO_Train_Global', ...
        %     {'RLAMPSO_Train', defaults.rlam.trainingEpisodes, defaults.rlam.trainedGlobalPath, defaults.rlam.configMode, 'global', defaults.popSize, defaults.maxIterations, defaults.rlam.initialW, defaults.rlam.initialC1, defaults.rlam.initialC2}, ...
        %     struct('requiresSerial', true));
        % makeAlgorithm('Train-OtherDRL', ...
        %     'RLAMPSO Training (5-Subgroup)', ...
        %     'RLAMPSO_Train_5Sub', ...
        %     {'RLAMPSO_Train', defaults.rlam.trainingEpisodes, defaults.rlam.trained5SubPath, defaults.rlam.configMode, '5subgroup', defaults.popSize, defaults.maxIterations, defaults.rlam.initialW, defaults.rlam.initialC1, defaults.rlam.initialC2}, ...
        %     struct('requiresSerial', true));
        % makeAlgorithm('Train-OtherDRL', ...
        %     'RLAMPSO Training (Per-Particle)', ...
        %     'RLAMPSO_Train_PerParticle', ...
        %     {'RLAMPSO_Train', defaults.rlam.trainingEpisodes, defaults.rlam.trainedPerParticlePath, defaults.rlam.configMode, 'per-particle', defaults.popSize, defaults.maxIterations, defaults.rlam.initialW, defaults.rlam.initialC1, defaults.rlam.initialC2}, ...
        %     struct('requiresSerial', true));

        % %--------------DQN-PSO Training (saves to models/DQN_PSO/)-----------------------%
        % makeAlgorithm('Train-OtherDRL', ...
        %     'DQN-PSO Training (Global)', ...
        %     'DQN_PSO_Train_Global', ...
        %     {'DQN_PSO_Train', defaults.dqn.trainingEpisodes, defaults.dqn.trainedGlobalPath, 'global', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2}, ...
        %     struct('requiresSerial', true));
        % makeAlgorithm('Train-OtherDRL', ...
        %     'DQN-PSO Training (5-Subgroup)', ...
        %     'DQN_PSO_Train_5Sub', ...
        %     {'DQN_PSO_Train', defaults.dqn.trainingEpisodes, defaults.dqn.trained5SubPath, '5subgroup', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2}, ...
        %     struct('requiresSerial', true));
        % makeAlgorithm('Train-OtherDRL', ...
        %     'DQN-PSO Training (Per-Particle)', ...
        %     'DQN_PSO_Train_PerParticle', ...
        %     {'DQN_PSO_Train', defaults.dqn.trainingEpisodes, defaults.dqn.trainedPerParticlePath, 'per-particle', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2}, ...
        %     struct('requiresSerial', true));
    };

    others = {
        makeAlgorithm('Others', ...
            'RRSACPSO', ...
            'RRSACPSO_Online', ...
            {'RRSACPSO_Online', defaults.maxIterations}, ...
            struct('requiresSerial', true));
        makeAlgorithm('Others', ...
            'PSO-Standard (Kennedy & Eberhart 1995)', ...
            'PSO_Standard', ...
            {'PSO', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2});
        makeAlgorithm('Others', ...
            'PSO-LDIW', ...
            'PSO_LDIW', ...
            {'PSO_LDIW', defaults.popSize, defaults.maxIterations, 0.9, defaults.standard.c1, defaults.standard.c2});
        makeAlgorithm('Others', ...
            'FIPS (Fully Informed PSO)', ...
            'FIPS', ...
            {'FIPS', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2});
        makeAlgorithm('Others', ...
            'CLPSO (Comprehensive Learning PSO)', ...
            'CLPSO', ...
            {'CLPSO', defaults.popSize, defaults.maxIterations, defaults.clpso.w, defaults.clpso.c});
        makeAlgorithm('Others', ...
            'HPSO-TVAC (Time-Varying Acceleration Coefficients)', ...
            'HPSO_TVAC', ...
            {'HPSO_TVAC', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2});
        % % makeAlgorithm('Others', ...
        % %     'DMS-PSO (Dynamic Multi-Swarm PSO)', ...
        % %     'DMS_PSO', ...
        % %     {'DMS_PSO', defaults.popSize, defaults.maxIterations, defaults.standard.w, defaults.standard.c1, defaults.standard.c2});
        % % makeAlgorithm('Others', ...
        % %     'LIPS (Learning PSO with Heterogeneous Interactions)', ...
        % %     'LIPS', ...
        % %     {'LIPS', defaults.popSize, defaults.maxIterations, defaults.lips.w, defaults.lips.c});
        makeAlgorithm('Others', ...
            'SAEPSO [11]', ...
            'SAEPSO_Full', ...
            {'SAEPSO', defaults.popSize, defaults.maxIterations, defaults.saepso.cMin, defaults.saepso.cMax, defaults.saepso.wMin, defaults.saepso.wMax, true});
        makeAlgorithm('Others', ...
            'SAEPSO* [11]', ...
            'SAEPSO_ParamsOnly', ...
            {'SAEPSO', defaults.popSize, defaults.maxIterations, defaults.saepso.cMin, defaults.saepso.cMax, defaults.saepso.wMin, defaults.saepso.wMax, false});
        makeAlgorithm('Others', ...
            'UAPSO [25]', ...
            'UAPSO', ...
            {'UAPSO', defaults.popSize, defaults.maxIterations, defaults.uapso.cMin, defaults.uapso.cMax});
    };

    % Configure group without editing this file repeatedly.
    % Supported values for VIETANH_ALGO_GROUP:
    %   rlbased (default), others, trainotherdrl, all
    groupSelection = lower(strtrim(getenv('VIETANH_ALGO_GROUP')));
    if isempty(groupSelection)
        groupSelection = 'rlbased';
    end

    switch groupSelection
        case 'rlbased'
            algorithms = rlBased;
        case 'others'
            algorithms = others;
        case 'trainotherdrl'
            algorithms = trainOtherDRL;
        case 'all'
            algorithms = dedupeAlgorithmsByFieldName({rlBased{:}, others{:}});
        otherwise
            error('Unknown VIETANH_ALGO_GROUP="%s". Use rlbased|others|trainotherdrl|all.', groupSelection);
    end

    algorithms = finaliseAlgorithms(algorithms);
end

% -------------------------------------------------------------------------
function defaults = defaultPsoParameters(paths)
    defaults.popSize = getenvOrDefault('VIETANH_POP_SIZE', 40);
    defaults.maxIterations = getenvOrDefault('VIETANH_MAX_ITERATIONS', 600);

    defaults.standard = struct('w', 0.7, 'c1', 1.5, 'c2', 1.5);
    defaults.dynamic  = struct('w', 0.7, 'c1', 1.5, 'c2', 1.5);
    defaults.clpso    = struct('w', 0.7298, 'c', 1.49445);
    defaults.lips     = struct('w', 0.7, 'c', 1.49);
    defaults.saepso   = struct('cMin', 0.5, 'cMax', 2.5, 'wMin', 0.2, 'wMax', 0.9);
    defaults.uapso    = struct('cMin', 0.0, 'cMax', 4.0);

    sharedTrainingEpisodes = 250;
    modelRoot = fullfile(paths.models, 'RRSACPSO');

    defaults.rlam = struct( ...
        'initialW',         0.7, ...
        'initialC1',        1.5, ...
        'initialC2',        1.5, ...
        'pretrainedNetwork','', ...
        'configMode',       'baseline', ...
        'trainingEpisodes', sharedTrainingEpisodes, ...
        'pretrainedGlobalPath', fullfile(modelRoot, 'rlampso_global.mat'), ...
        'pretrained5SubPath', fullfile(modelRoot, 'rlampso_5sub.mat'), ...
        'pretrainedPerParticlePath', fullfile(modelRoot, 'rlampso_perparticle.mat'), ...
        'trainedGlobalPath', fullfile(paths.models, 'RLAMPSO', 'rlampso_global_trained.mat'), ...
        'trained5SubPath', fullfile(paths.models, 'RLAMPSO', 'rlampso_5sub_trained.mat'), ...
        'trainedPerParticlePath', fullfile(paths.models, 'RLAMPSO', 'rlampso_perparticle_trained.mat') ...
    );

    defaults.apex = struct('mode', 'online_only');

    defaults.dqn = struct( ...
        'trainingEpisodes', sharedTrainingEpisodes, ...
        'pretrainedGlobalPath', fullfile(modelRoot, 'dqn_pso_global.mat'), ...
        'pretrained5SubPath', fullfile(modelRoot, 'dqn_pso_5sub.mat'), ...
        'pretrainedPerParticlePath', fullfile(modelRoot, 'dqn_pso_perparticle.mat'), ...
        'trainedGlobalPath', fullfile(paths.models, 'DQN_PSO', 'dqn_pso_global_trained.mat'), ...
        'trained5SubPath', fullfile(paths.models, 'DQN_PSO', 'dqn_pso_5sub_trained.mat'), ...
        'trainedPerParticlePath', fullfile(paths.models, 'DQN_PSO', 'dqn_pso_perparticle_trained.mat'), ...
        'alpha', 0.1, ...
        'gamma', 0.9, ...
        'epsilon', 0.2 ...
    );
end

% -------------------------------------------------------------------------
function algorithm = makeAlgorithm(category, displayName, fieldName, specificParams, opts)
    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    algorithm = struct();
    algorithm.category = category;
    algorithm.function = @(varargin) directGlobalPlanning(varargin{:});  % Pure PSO optimization (no hierarchical planning)
    algorithm.displayName = displayName;
    algorithm.fieldName = fieldName;
    algorithm.specificParams = specificParams;

    if isfield(opts, 'requiresSerial')
        algorithm.requiresSerial = logical(opts.requiresSerial);
    end
end

% -------------------------------------------------------------------------
function algorithms = finaliseAlgorithms(algorithms)
    for idx = 1:numel(algorithms)
        if ~isfield(algorithms{idx}, 'requiresSerial')
            algorithms{idx}.requiresSerial = false;
        end
        algorithms{idx}.seedOffset = idx;
    end
end

% -------------------------------------------------------------------------
function algorithms = dedupeAlgorithmsByFieldName(algorithms)
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    keepMask = true(1, numel(algorithms));

    for idx = 1:numel(algorithms)
        fieldName = char(algorithms{idx}.fieldName);
        if isKey(seen, fieldName)
            keepMask(idx) = false;
        else
            seen(fieldName) = true;
        end
    end

    algorithms = algorithms(keepMask);
end

% -------------------------------------------------------------------------
function ensureDirectory(directoryPath)
    if exist(directoryPath, 'dir') ~= 7
        mkdir(directoryPath);
        fprintf('Created directory: %s\n', directoryPath);
    end
end

% -------------------------------------------------------------------------
function logFile = startLogging(logDir)
    timestamp = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
    logFile = fullfile(logDir, sprintf('run_comparison_%s.log', char(timestamp)));
    diary(logFile);
end

% -------------------------------------------------------------------------
function reportFailure(exception, logFile)
    fprintf('\nERROR OCCURRED\n');
    fprintf('Log file: %s\n', logFile);
    fprintf('Error: %s\n', exception.message);
    stack = exception.stack;
    for idx = 1:numel(stack)
        fprintf('  %d. %s (line %d)\n', idx, stack(idx).name, stack(idx).line);
    end
    fprintf('\n');
end

% -------------------------------------------------------------------------
function stopDiary()
    try
        diary('off');
    catch
        % Ignore failures to close diary (already closed or not started)
    end
end

function scenarios = filterScenarioSet(scenarios)
    scenarioFilter = lower(strtrim(getenv('VIETANH_SCENARIO_FILTER')));
    if isempty(scenarioFilter)
        return;
    end

    labels = arrayfun(@(s) lower(string(s.label)), scenarios, 'UniformOutput', false);
    keepMask = false(size(scenarios));
    for idx = 1:numel(scenarios)
        keepMask(idx) = contains(labels{idx}, scenarioFilter);
    end

    if ~any(keepMask)
        error('VIETANH_SCENARIO_FILTER="%s" matched no scenarios.', scenarioFilter);
    end
    scenarios = scenarios(keepMask);
end

function value = getenvOrDefault(name, defaultValue)
    rawValue = strtrim(getenv(name));
    if isempty(rawValue)
        value = defaultValue;
        return;
    end

    parsed = str2double(rawValue);
    if ~isfinite(parsed)
        error('Environment variable %s="%s" is not numeric.', name, rawValue);
    end
    value = parsed;
end

function value = getenvStringOrDefault(name, defaultValue)
    rawValue = strtrim(getenv(name));
    if isempty(rawValue)
        value = defaultValue;
    else
        value = rawValue;
    end
end
