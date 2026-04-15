function config = PPOPSO_Config(mode)
    % PPOPSO configuration adapted from Klein et al. 2024 (iSOMA-RL) to PSO.
    %
    % Continuous PPO control of PSO parameters (w, c1, c2).
    % State: FE completion + history of fitness, fitness diff, and actions.
    % Online learning (no pretraining), following the paper's protocol.

    if nargin < 1
        mode = 'online';
    end

    config = struct();
    config.mode = mode;
    config.algorithm = 'PPO-PSO';
    config.version = '2.0';


    % ===== STATE DESIGN (adapted from Klein 2024 Table 5) =====
    % Paper uses: fitness history, fitness diff history, action history
    % hl=25 for 50000 FEs; we scale to hl=5 for 1000 iterations
    config.historyLen = 5;
    config.actionDim = 3;  % w, c1, c2
    % State: [FE_norm, fitness(hl), fitDiff(hl), actions(hl*3)]
    config.stateSize = 1 + config.historyLen * (1 + 1 + config.actionDim);

    % ===== NETWORK ARCHITECTURE (Klein 2024 Table 4) =====
    config.actorHiddenLayers = [64, 64];
    config.criticHiddenLayers = [64, 64];

    % ===== PPO HYPERPARAMETERS (Klein 2024 Table 3) =====
    config.gamma = 0.99;
    config.gaeLambda = 0.95;
    config.clipEpsilon = 0.2;
    config.entropyCoef = 0.0;    % Paper default: 0.0
    config.valueCoef = 0.5;
    config.actorLR = 3e-4;
    config.criticLR = 3e-4;
    config.maxGradNorm = 0.5;

    % ===== TRAINING CONFIGURATION =====
    config.numEpisodes = 1;          % Online learning (paper: no pretraining)
    config.maxIterations = 1000;
    config.updateInterval = 64;      % PPO rollout buffer size (paper: n_steps=2048, scaled)
    config.updateEpochs = 10;        % Paper: n_epochs=10
    config.minibatchSize = 64;       % Paper: batch_size=64

    % ===== PSO CONFIGURATION =====
    config.popSize = 100;
    config.numWaypoints = 5;
    config.mapSize = [400, 400, 100];

    % ===== PARAMETER RANGES =====
    config.wMin = 0.1;   config.wMax = 0.9;
    config.c1Min = 0.5;  config.c1Max = 2.5;
    config.c2Min = 0.5;  config.c2Max = 2.5;

    % ===== VELOCITY CLAMPING (per-dimension, same as other methods) =====
    config.velocityClampDelta = 0.5;

    % ===== GAUSSIAN POLICY =====
    config.initLogStd = -0.5;  % Initial log std for action noise

    % ===== LOGGING =====
    config.logInterval = 50;
    config.verbose = false;
    config.disableVisualization = false;
end
