function config = RLAMPSO_Config(mode)
    %RLAMPSO_CONFIG Configuration for the retained RLAMPSO implementation.

    if nargin < 1 || isempty(mode)
        mode = 'baseline';
    end

    config = struct();

    % Bookkeeping
    config.mode = lower(mode);

    % Paper-aligned default controller: 5 subgroups, 4 outputs each.
    config.stateSize = 15;
    config.actionSize = 20;
    config.numSubgroups = 5;
    config.paramMode = '5subgroup';

    % Network shape used by the retained pretrained checkpoint.
    config.actorHiddenLayers = [64, 64, 64];
    config.criticHiddenLayers = [64, 64, 32, 32, 16];

    % DDPG hyperparameters.
    config.actorLR = 1e-4;
    config.criticLR = 1e-3;
    config.gamma = 0.99;
    config.tau = 1e-3;
    config.explorationNoise = 0.5;
    config.batchSize = 256;
    config.bufferSize = 100000;
    config.rewardType = 'binary';

    % Generic benchmark defaults. Harnesses override these when needed.
    config.numWaypoints = 5;
    config.mapSize = [400, 400, 100];
    config.popSize = 40;
    config.maxIterations = 600;
    config.numEpisodes = 100;
    config.trainEveryNIterations = 1;
    config.trainDuringDeployment = true;

    % Runtime and logging.
    config.useGPU = false;
    config.verbose = false;
    config.logInterval = 5;
    config.saveInterval = 100;
    config.savePath = 'models/rlampso/rlampso.mat';

    switch config.mode
        case 'baseline'
            % Keep defaults.

        case 'pretrain'
            % Match the SAC benchmark pretraining budget.
            config.popSize = 100;
            config.maxIterations = 1000;
            config.numEpisodes = 5000;
            config.bufferSize = 150000;
            config.saveInterval = 50;
            config.savePath = 'models/rlampso/pretrained_rlampso.mat';

        case 'fast'
            config.popSize = 30;
            config.maxIterations = 300;
            config.numEpisodes = 20;

        otherwise
            error('RLAMPSO_Config:InvalidMode', 'Unsupported mode "%s".', mode);
    end
end
