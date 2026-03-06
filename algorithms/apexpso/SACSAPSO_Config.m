function config = SACSAPSO_Config(mode)
    % SAC-SAPSO configuration aligned with the paper (Mathematics 2024, 12, 3481).

    if nargin < 1
        mode = 'paper';
    end

    config = struct();
    config.mode = mode;
    config.algorithm = 'SAC-SAPSO';
    config.version = 'paper-2024';
    config.createdAt = datetime('now');

    % Core SAC settings (paper-aligned)
    config.useSAC = true;
    config.useAutomaticEntropyTuning = true;
    config.useBatchNorm = false;
    config.useTargetNetworks = true;
    config.numCritics = 2;

    % State/action configuration (global CPs)
    config.usePerParticleActions = false;
    config.useMultiObjectiveReward = false;
    config.useCuriosityBonus = false;
    config.paramsPerParticle = 3;
    config.popSize = 30;
    config.numWaypoints = 5;
    config.actionSize = 3;
    config.stateSize = config.popSize + 3;

    % Network architecture (SAC defaults)
    config.actorHiddenLayers = [256, 256];
    config.criticHiddenLayers = [256, 256];

    % SAC hyperparameters (Table 2)
    config.actorLR = 1e-4;
    config.criticLR = 1e-4;
    config.alphaLR = 1e-4;
    config.gamma = 1.0;
    config.tau = 0.005;
    config.targetEntropy = -config.actionSize;
    config.initAlpha = 0.2;

    % Training configuration
    config.numEpisodes = 1;
    config.maxIterations = 600;
    config.trainEveryNIterations = 1;
    config.gradientStepsPerTraining = 1;
    config.batchSize = 32;
    config.bufferSize = 1e6;
    config.warmupPeriod = 0;
    config.explorationNoiseStart = 0.0;
    config.explorationNoiseEnd = 0.0;
    config.explorationDecay = 1.0;

    % Observation interval (nt)
    config.observationInterval = 125;

    % Parameter ranges
    config.paramRanges = struct( ...
        'wMin', 0.1, 'wMax', 0.9, ...
        'c1Min', 0.5, 'c1Max', 2.5, ...
        'c2Min', 0.5, 'c2Max', 2.5);

    % Map and logging defaults
    config.mapSize = [400, 400, 100];
    config.logInterval = 50;
    config.saveInterval = 999999;
    config.savePath = 'models/sac_sapso_paper.mat';
    config.verbose = false;
    config.modelPath = '';

    % BatchNorm defaults (unused when disabled)
    config.batchNormMomentum = 0.99;
    config.batchNormEpsilon = 1e-5;

    % GPU configuration
    config.useGPU = false;
    config.gpuName = 'None';
    config.gpuMemory = 0;
end
