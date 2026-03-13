function td3Agent = initializeDLToolboxTD3(stateSize, actionSize, usePER, pretrainedPath, verbose)
    % Initialize TD3 Agent using MATLAB Deep Learning Toolbox
    %
    % Modern replacement for initializeTD3Agent.m
    % Uses dlnetwork for twin critics, automatic differentiation, GPU acceleration
    %
    % TD3 Improvements over DDPG:
    %   1. Twin Q-networks (Clipped Double Q-learning)
    %   2. Delayed policy updates (update actor less frequently)
    %   3. Target policy smoothing (add noise to target actions)
    %
    % Inputs:
    %   stateSize: Dimension of state space (default: 15)
    %   actionSize: Dimension of action space (default: 20)
    %   usePER: Use Prioritized Experience Replay (default: true)
    %   pretrainedPath: Path to pretrained networks (optional)
    %   verbose: Print detailed output (default: false)
    %
    % Outputs:
    %   td3Agent: TD3 agent structure with dlnetwork twin critics
    %
    % Usage:
    %   agent = initializeDLToolboxTD3(15, 20, true);   % Baseline TD3 + PER
    %   agent = initializeDLToolboxTD3(64, 20, true);   % Transformer state + PER

    if nargin < 1, stateSize = 15; end
    if nargin < 2, actionSize = 20; end
    if nargin < 3, usePER = true; end
    if nargin < 4, pretrainedPath = ''; end
    if nargin < 5, verbose = false; end

    if verbose
        fprintf('\n╔════════════════════════════════════════════╗\n');
        fprintf('║  Initializing TD3 Agent (DL Toolbox)     ║\n');
        fprintf('╚════════════════════════════════════════════╝\n\n');
    end

    td3Agent = struct();

    % Network dimensions
    td3Agent.stateSize = stateSize;
    td3Agent.actionSize = actionSize;
    td3Agent.numSubgroups = 5;  % For PSO parameter conversion

    % TD3 hyperparameters
    td3Agent.gamma = 0.99;
    td3Agent.tau = 0.005;  % Soft update rate (higher than DDPG for stability)
    td3Agent.actorLR = 0.0001;   % 1e-4
    td3Agent.criticLR = 0.001;   % 1e-3
    td3Agent.batchSize = 256;
    td3Agent.bufferSize = 1000000;  % 1M experiences
    td3Agent.trainStep = 0;

    % Learning rate warmup (prevents early divergence)
    td3Agent.warmupSteps = 1000;           % Linear warmup for 1000 steps
    td3Agent.initialActorLR = 0.00001;     % 10x lower initial LR (1e-5)
    td3Agent.initialCriticLR = 0.001;      % 1e-3 (FIXED: matches DDPG pattern and Python reference)

    % TD3-specific parameters
    td3Agent.policyDelay = 2;           % Update actor every 2 critic updates
    td3Agent.targetNoiseStd = 0.2;      % Target policy smoothing noise
    td3Agent.targetNoiseClip = 0.5;     % Noise clip range

    % Exploration noise (decaying)
    td3Agent.explorationNoise = 0.1;    % Lower initial noise than DDPG
    td3Agent.noiseDecay = 0.995;
    td3Agent.minNoise = 0.01;

    % GPU detection
    td3Agent.useGPU = canUseGPU();
    td3Agent.useDLToolbox = true;  % Flag for compatibility
    td3Agent.useTD3 = true;        % Flag to distinguish from DDPG

    % Prioritized Experience Replay
    td3Agent.usePER = usePER;

    if usePER
        alpha = 0.6;
        beta_start = 0.4;
        beta_frames = 100000;
        td3Agent.replayBuffer = PrioritizedReplayBuffer(td3Agent.bufferSize, ...
                                                       alpha, beta_start, beta_frames);
        if verbose
            fprintf('✓ Prioritized Experience Replay enabled\n');
        end
    else
        td3Agent.replayBuffer = [];
        td3Agent.bufferIndex = 1;
        if verbose
            fprintf('✓ Standard Experience Replay enabled\n');
        end
    end

    % Initialize or load networks
    if ~isempty(pretrainedPath) && exist(pretrainedPath, 'file')
        if verbose
            fprintf('Loading pretrained TD3 networks from: %s\n', pretrainedPath);
        end
        loaded = load(pretrainedPath);

        % Load networks (assuming they're already dlnetwork)
        td3Agent.actor = loaded.actor;
        td3Agent.critic1 = loaded.critic1;
        td3Agent.critic2 = loaded.critic2;
        td3Agent.targetActor = loaded.targetActor;
        td3Agent.targetCritic1 = loaded.targetCritic1;
        td3Agent.targetCritic2 = loaded.targetCritic2;

        if verbose
            fprintf('✓ Pretrained twin critic networks loaded\n');
        end
    else
        % Build new networks using DL Toolbox
        if verbose
            fprintf('Building new TD3 networks (twin critics)...\n');
        end

        td3Agent.actor = buildActorNetwork(stateSize, actionSize, verbose);
        td3Agent.critic1 = buildCriticNetwork(stateSize, actionSize, verbose);
        td3Agent.critic2 = buildCriticNetwork(stateSize, actionSize, verbose);  % Twin critic

        % Initialize target networks as copies
        td3Agent.targetActor = td3Agent.actor;
        td3Agent.targetCritic1 = td3Agent.critic1;
        td3Agent.targetCritic2 = td3Agent.critic2;

        if verbose
            fprintf('✓ New twin critic networks initialized\n');
        end
    end

    % ===== MOVE NETWORKS TO GPU (CRITICAL FOR GPU USAGE) =====
    if td3Agent.useGPU && canUseGPU()
        fprintf('Moving TD3 networks to GPU...\n');

        % Move all networks to GPU - proper method for dlnetwork
        td3Agent.actor = moveNetworkToGPU(td3Agent.actor);
        td3Agent.critic1 = moveNetworkToGPU(td3Agent.critic1);
        td3Agent.critic2 = moveNetworkToGPU(td3Agent.critic2);
        td3Agent.targetActor = moveNetworkToGPU(td3Agent.targetActor);
        td3Agent.targetCritic1 = moveNetworkToGPU(td3Agent.targetCritic1);
        td3Agent.targetCritic2 = moveNetworkToGPU(td3Agent.targetCritic2);

        fprintf('✓ All TD3 networks moved to GPU\n');
    end

    % Initialize optimizers (Adam for actor and both critics)
    td3Agent.actorOptimizer = struct();
    td3Agent.actorOptimizer.averageGrad = [];
    td3Agent.actorOptimizer.averageSqGrad = [];
    td3Agent.actorOptimizer.gradDecay = 0.9;
    td3Agent.actorOptimizer.sqGradDecay = 0.999;

    td3Agent.critic1Optimizer = struct();
    td3Agent.critic1Optimizer.averageGrad = [];
    td3Agent.critic1Optimizer.averageSqGrad = [];
    td3Agent.critic1Optimizer.gradDecay = 0.9;
    td3Agent.critic1Optimizer.sqGradDecay = 0.999;

    td3Agent.critic2Optimizer = struct();
    td3Agent.critic2Optimizer.averageGrad = [];
    td3Agent.critic2Optimizer.averageSqGrad = [];
    td3Agent.critic2Optimizer.gradDecay = 0.9;
    td3Agent.critic2Optimizer.sqGradDecay = 0.999;

    % State tracking
    td3Agent.lastState = [];
    td3Agent.lastAction = [];

    % Display configuration (only if verbose)
    if verbose
        fprintf('\n=== TD3 Configuration (DL Toolbox) ===\n');
        fprintf('State Size: %d | Action Size: %d\n', stateSize, actionSize);
        fprintf('Actor LR: %.5f | Critic LR: %.5f\n', td3Agent.actorLR, td3Agent.criticLR);
        fprintf('Gamma: %.3f | Tau: %.4f\n', td3Agent.gamma, td3Agent.tau);
        fprintf('Policy Delay: %d | Target Noise: %.2f (clip ±%.2f)\n', ...
                td3Agent.policyDelay, td3Agent.targetNoiseStd, td3Agent.targetNoiseClip);
        fprintf('Batch Size: %d | Buffer: %d\n', td3Agent.batchSize, td3Agent.bufferSize);
        fprintf('PER: %s | GPU: %s\n', ...
                boolToStr(td3Agent.usePER), boolToStr(td3Agent.useGPU));
        fprintf('======================================\n\n');

        fprintf('✓ TD3 Agent (DL Toolbox) initialized successfully!\n\n');
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
