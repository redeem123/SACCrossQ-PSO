function ddpgAgent = initializeDLToolboxDDPG(stateSize, actionSize, ~, pretrainedPath, verbose)
    % Initialize DDPG Agent using MATLAB Deep Learning Toolbox
    %
    % Paper: Yin et al., 2023 — 4 networks (actor, target actor, critic, target critic)
    %
    % Inputs:
    %   stateSize:      State dimension (15 for paper baseline)
    %   actionSize:     Action dimension (20 for 5 subgroups × 4 params)
    %   ~:              Unused (legacy PER flag)
    %   pretrainedPath: Path to pretrained networks (optional)
    %   verbose:        Print output (default: false)

    if nargin < 1, stateSize  = 15;    end
    if nargin < 2, actionSize = 20;    end
    if nargin < 4, pretrainedPath = ''; end
    if nargin < 5, verbose = false;     end

    ddpgAgent = struct();

    % Dimensions
    ddpgAgent.stateSize    = stateSize;
    ddpgAgent.actionSize   = actionSize;
    ddpgAgent.numSubgroups = 5;

    % DDPG hyperparameters (paper Section 3.2)
    ddpgAgent.gamma    = 0.99;
    ddpgAgent.tau      = 0.001;   % Soft update τ << 1 (Eq. 1)
    ddpgAgent.actorLR  = 1e-4;
    ddpgAgent.criticLR = 1e-3;
    ddpgAgent.batchSize  = 256;
    ddpgAgent.bufferSize = 100000;
    ddpgAgent.trainStep  = 0;

    % CPU-only
    ddpgAgent.useGPU      = false;
    ddpgAgent.useDLToolbox = true;


    % Exploration noise: N(0, 0.5) per paper Eq. 14, constant (no decay)
    ddpgAgent.explorationNoise = 0.5;

    % Uniform replay buffer
    ddpgAgent.replayBuffer = [];
    ddpgAgent.bufferIndex  = 1;

    % Build or load networks
    if ~isempty(pretrainedPath) && exist(pretrainedPath, 'file')
        loaded = load(pretrainedPath);
        ddpgAgent.actor        = loaded.actor;
        ddpgAgent.critic       = loaded.critic;
        ddpgAgent.targetActor  = loaded.targetActor;
        ddpgAgent.targetCritic = loaded.targetCritic;
        if verbose, fprintf('Loaded pretrained DDPG from: %s\n', pretrainedPath); end
    else
        % Paper Figs. 6–7: actor 4-layer FC, critic 6-layer FC
        ddpgAgent.actor  = buildActorNetwork(stateSize, actionSize, verbose);
        ddpgAgent.critic = buildCriticNetwork(stateSize, actionSize, verbose);
        ddpgAgent.targetActor  = ddpgAgent.actor;
        ddpgAgent.targetCritic = ddpgAgent.critic;
    end

    % Adam optimizers
    ddpgAgent.actorOptimizer  = struct('averageGrad',[],'averageSqGrad',[],...
                                       'gradDecay',0.9,'sqGradDecay',0.999);
    ddpgAgent.criticOptimizer = struct('averageGrad',[],'averageSqGrad',[],...
                                       'gradDecay',0.9,'sqGradDecay',0.999);

    % State tracking
    ddpgAgent.lastState  = [];
    ddpgAgent.lastAction = [];

    if verbose
        fprintf('DDPG Agent initialized: state=%d, action=%d, tau=%.4f\n', ...
                stateSize, actionSize, ddpgAgent.tau);
    end
end
