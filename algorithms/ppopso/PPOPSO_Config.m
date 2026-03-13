function config = PPOPSO_Config(mode)
    % PPOPSO configuration aligned to the PPOPSO paper.

    if nargin < 1
        mode = 'online';
    end

    config = struct();
    config.mode = mode;
    config.algorithm = 'PPO-PSO';
    config.version = '1.1';
    config.createdAt = datetime('now');

    % ===== PPOPSO PAPER SETTINGS =====
    config.numSubgroups = 4;
    config.historyLen = 5;
    config.numActionConfigs = 5;
    config.stateSize = 1 + config.historyLen * (1 + config.numSubgroups);

    config.actorHiddenLayers = [64, 64];
    config.criticHiddenLayers = [64, 64];

    % ===== PPO HYPERPARAMETERS =====
    config.gamma = 0.99;
    config.gaeLambda = 0.95;
    config.clipEpsilon = 0.2;
    config.entropyCoef = 0.01;
    config.valueCoef = 0.5;
    config.actorLR = 3e-4;
    config.criticLR = 3e-4;
    config.maxGradNorm = 1.0;

    % ===== TRAINING CONFIGURATION =====
    config.numEpisodes = 1;
    config.maxIterations = 600;
    config.updateInterval = 64;
    config.updateEpochs = 10;
    config.minibatchSize = 64;

    % ===== PSO CONFIGURATION =====
    config.popSize = 100;
    config.numWaypoints = 5;
    config.mapSize = [400, 400, 100];
    config.paramsPerParticle = 3;
    config.migrationInterval = 10;
    config.migrationRate = 0.1;
    config.velocityClampFactor = 0.5;

    % ===== STATE/REWARD SETTINGS =====
    config.historyInitValue = 1e9;
    config.maxFunctionEvals = [];

    % ===== ACTION CONFIGURATIONS (TABLE 1) =====
    config.actionTable = [
        0.729, 1.49445, 1.49445;  % Standard PSO parameters
        0.9,   1.8,     1.2;      % Strong exploration
        0.4,   1.2,     1.8;      % Strong exploitation
        0.9,   1.5,     1.5;      % Linearly decreasing inertia weight
        0.6,   2.0,     1.0       % Another parameter variant
    ];
    config.linearWActionIndex = 4;
    config.linearWStart = 0.9;
    config.linearWEnd = 0.4;

    % ===== LOGGING =====
    config.logInterval = 50;
    config.verbose = false;
end
