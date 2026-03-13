function ddpgAgent = initializeDLToolboxDDPG(stateSize, actionSize, usePER, pretrainedPath, verbose)
    % Initialize DDPG Agent using MATLAB Deep Learning Toolbox
    %
    % Modern replacement for initializePaperExactDDPGAgent.m
    % Uses dlnetwork for automatic differentiation and GPU acceleration
    %
    % Inputs:
    %   stateSize: Dimension of state space (default: 15)
    %   actionSize: Dimension of action space (default: 20)
    %   usePER: Use Prioritized Experience Replay (default: false)
    %   pretrainedPath: Path to pretrained networks (optional)
    %   verbose: Print detailed output (default: false)
    %
    % Outputs:
    %   ddpgAgent: DDPG agent structure with dlnetwork networks
    %
    % Usage:
    %   agent = initializeDLToolboxDDPG(15, 20, false);  % Baseline DDPG
    %   agent = initializeDLToolboxDDPG(64, 20, true);   % With PER & Transformer

    if nargin < 1, stateSize = 15; end
    if nargin < 2, actionSize = 20; end
    if nargin < 3, usePER = false; end
    if nargin < 4, pretrainedPath = ''; end
    if nargin < 5, verbose = false; end

    if verbose
        fprintf('\n╔════════════════════════════════════════════╗\n');
        fprintf('║  Initializing DDPG Agent (DL Toolbox)    ║\n');
        fprintf('╚════════════════════════════════════════════╝\n\n');
    end

    ddpgAgent = struct();

    % Network dimensions
    ddpgAgent.stateSize = stateSize;
    ddpgAgent.actionSize = actionSize;
    ddpgAgent.numSubgroups = 5;  % For PSO parameter conversion

    % DDPG hyperparameters
    ddpgAgent.gamma = 0.99;
    ddpgAgent.tau = 0.001;  % Soft update rate
    ddpgAgent.actorLR = 0.0001;   % 1e-4
    ddpgAgent.criticLR = 0.001;   % 1e-3
    ddpgAgent.batchSize = 256;
    ddpgAgent.bufferSize = 1000000;  % 1M experiences
    ddpgAgent.trainStep = 0;

    % Learning rate warmup (prevents early divergence)
    ddpgAgent.warmupSteps = 1000;           % Linear warmup for 1000 steps
    ddpgAgent.initialActorLR = 0.00001;     % 1e-5 (matches Python TF2_DDPG_Basic.py:99)
    ddpgAgent.initialCriticLR = 0.001;      % 1e-3 (FIXED: matches Python TF2_DDPG_Basic.py:100)

    % GPU detection
    ddpgAgent.useGPU = canUseGPU();
    ddpgAgent.useDLToolbox = true;  % Flag for compatibility

    % Exploration noise (decaying)
    ddpgAgent.explorationNoise = 0.2;    % Higher initial noise for DDPG
    ddpgAgent.noiseDecay = 0.995;
    ddpgAgent.minNoise = 0.01;

    % Prioritized Experience Replay
    ddpgAgent.usePER = usePER;

    if usePER
        alpha = 0.6;
        beta_start = 0.4;
        beta_frames = 100000;
        ddpgAgent.replayBuffer = PrioritizedReplayBuffer(ddpgAgent.bufferSize, ...
                                                        alpha, beta_start, beta_frames);
        if verbose
            fprintf('✓ Prioritized Experience Replay enabled\n');
        end
    else
        ddpgAgent.replayBuffer = [];
        ddpgAgent.bufferIndex = 1;
        if verbose
            fprintf('✓ Standard Experience Replay enabled\n');
        end
    end

    % Initialize or load networks
    if ~isempty(pretrainedPath) && exist(pretrainedPath, 'file')
        if verbose
            fprintf('Loading pretrained networks from: %s\n', pretrainedPath);
        end
        loaded = load(pretrainedPath);

        % Load networks (assuming they're already dlnetwork)
        ddpgAgent.actor = loaded.actor;
        ddpgAgent.critic = loaded.critic;
        ddpgAgent.targetActor = loaded.targetActor;
        ddpgAgent.targetCritic = loaded.targetCritic;

        if verbose
            fprintf('✓ Pretrained networks loaded\n');
        end
    else
        % Build new networks using DL Toolbox
        if verbose
            fprintf('Building new DDPG networks...\n');
        end

        ddpgAgent.actor = buildActorNetwork(stateSize, actionSize, verbose);
        ddpgAgent.critic = buildCriticNetwork(stateSize, actionSize, verbose);

        % Initialize target networks as copies
        ddpgAgent.targetActor = ddpgAgent.actor;
        ddpgAgent.targetCritic = ddpgAgent.critic;

        if verbose
            fprintf('✓ New networks initialized\n');
        end
    end

    % ===== MOVE NETWORKS TO GPU (CRITICAL FOR GPU USAGE) =====
    if ddpgAgent.useGPU && canUseGPU()
        fprintf('Moving DDPG networks to GPU...\n');

        % Move all networks to GPU - proper method for dlnetwork
        ddpgAgent.actor = moveNetworkToGPU(ddpgAgent.actor);
        ddpgAgent.critic = moveNetworkToGPU(ddpgAgent.critic);
        ddpgAgent.targetActor = moveNetworkToGPU(ddpgAgent.targetActor);
        ddpgAgent.targetCritic = moveNetworkToGPU(ddpgAgent.targetCritic);

        fprintf('✓ All DDPG networks moved to GPU\n');
    end

    % Initialize optimizers (Adam for both actor and critic)
    ddpgAgent.actorOptimizer = struct();
    ddpgAgent.actorOptimizer.averageGrad = [];
    ddpgAgent.actorOptimizer.averageSqGrad = [];
    ddpgAgent.actorOptimizer.gradDecay = 0.9;
    ddpgAgent.actorOptimizer.sqGradDecay = 0.999;

    ddpgAgent.criticOptimizer = struct();
    ddpgAgent.criticOptimizer.averageGrad = [];
    ddpgAgent.criticOptimizer.averageSqGrad = [];
    ddpgAgent.criticOptimizer.gradDecay = 0.9;
    ddpgAgent.criticOptimizer.sqGradDecay = 0.999;

    % State tracking
    ddpgAgent.lastState = [];
    ddpgAgent.lastAction = [];

    % Display configuration (only if verbose)
    if verbose
        fprintf('\n=== DDPG Configuration (DL Toolbox) ===\n');
        fprintf('State Size: %d | Action Size: %d\n', stateSize, actionSize);
        fprintf('Actor LR: %.5f | Critic LR: %.5f\n', ddpgAgent.actorLR, ddpgAgent.criticLR);
        fprintf('Gamma: %.3f | Tau: %.4f\n', ddpgAgent.gamma, ddpgAgent.tau);
        fprintf('Batch Size: %d | Buffer: %d\n', ddpgAgent.batchSize, ddpgAgent.bufferSize);
        fprintf('PER: %s | GPU: %s\n', ...
                boolToStr(ddpgAgent.usePER), boolToStr(ddpgAgent.useGPU));
        fprintf('=======================================\n\n');

        fprintf('✓ DDPG Agent (DL Toolbox) initialized successfully!\n\n');
    end
end

function str = boolToStr(value)
    if value
        str = 'ON';
    else
        str = 'OFF';
    end
end

function net = moveNetworkToGPU(net)
    % Properly move dlnetwork to GPU by converting all learnables
    % This is the correct way to move dlnetwork to GPU in MATLAB

    learnables = net.Learnables;
    for i = 1:height(learnables)
        learnables.Value{i} = gpuArray(learnables.Value{i});
    end
    net.Learnables = learnables;
end
