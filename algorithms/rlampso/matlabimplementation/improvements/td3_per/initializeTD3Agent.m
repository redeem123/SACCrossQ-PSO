function td3Agent = initializeTD3Agent(stateSize, actionSize, usePER, pretrainedPath)
    % Initialize TD3 (Twin Delayed DDPG) Agent with optional PER
    % Replaces initializePaperExactDDPGAgent.m with TD3 improvements
    %
    % Reference: Fujimoto et al., "Addressing Function Approximation Error in
    %            Actor-Critic Methods", ICML 2018
    %
    % TD3 Key Improvements over DDPG:
    %   1. Twin Q-networks (Clipped Double Q-learning)
    %   2. Delayed policy updates (update actor less frequently)
    %   3. Target policy smoothing (add noise to target actions)
    %
    % Inputs:
    %   stateSize: Dimension of state space (15 for sin-encoding, 64 for transformer)
    %   actionSize: Dimension of action space (default: 20)
    %   usePER: Use Prioritized Experience Replay (default: true)
    %   pretrainedPath: (optional) Path to pretrained networks
    %
    % Outputs:
    %   td3Agent: Agent structure with twin critics and TD3 parameters

    if nargin < 1, stateSize = 15; end
    if nargin < 2, actionSize = 20; end
    if nargin < 3, usePER = true; end
    if nargin < 4, pretrainedPath = ''; end

    td3Agent = struct();

    % Network dimensions
    td3Agent.stateSize = stateSize;
    td3Agent.actionSize = actionSize;
    td3Agent.numSubgroups = 5;  % For PSO parameter conversion

    % === TD3 HYPERPARAMETERS ===

    % Learning rates
    td3Agent.actorLR = 0.0001;   % 1e-4
    td3Agent.criticLR = 0.001;   % 1e-3

    % Discount and soft update
    td3Agent.gamma = 0.99;
    td3Agent.tau = 0.005;  % Soft update rate

    % TD3-specific parameters
    td3Agent.policyDelay = 2;           % Update actor every 2 critic updates
    td3Agent.targetNoiseStd = 0.2;      % Std of target policy smoothing noise
    td3Agent.targetNoiseClip = 0.5;     % Clip range for target noise
    td3Agent.explorationNoise = 0.1;    % Exploration noise std (decays over time)

    % Training parameters
    td3Agent.batchSize = 256;
    td3Agent.bufferSize = 1000000;  % 1M experiences
    td3Agent.trainStep = 0;

    % === PRIORITIZED EXPERIENCE REPLAY ===
    td3Agent.usePER = usePER;

    if usePER
        % PER parameters
        alpha = 0.6;        % Priority exponent
        beta_start = 0.4;   % Initial importance sampling weight
        beta_frames = 100000;  % Frames to anneal beta to 1.0

        td3Agent.replayBuffer = PrioritizedReplayBuffer(td3Agent.bufferSize, ...
                                                        alpha, beta_start, beta_frames);
        fprintf('Prioritized Experience Replay enabled (alpha=%.2f, beta=%.2f->1.0)\n', ...
                alpha, beta_start);
    else
        % Standard replay buffer
        td3Agent.replayBuffer = [];
        td3Agent.bufferIndex = 1;
        fprintf('Standard Experience Replay enabled\n');
    end

    % GPU detection
    td3Agent.useGPU = canUseGPU();
    if td3Agent.useGPU
        fprintf('GPU detected and enabled\n');
    end

    % === INITIALIZE NETWORKS ===

    if ~isempty(pretrainedPath) && exist(pretrainedPath, 'file')
        % Load pretrained networks
        fprintf('Loading pretrained TD3 networks from: %s\n', pretrainedPath);
        loaded = load(pretrainedPath);

        td3Agent.actor = loaded.trainedActor;
        td3Agent.critic1 = loaded.trainedCritic1;
        td3Agent.critic2 = loaded.trainedCritic2;
        td3Agent.targetActor = loaded.trainedTargetActor;
        td3Agent.targetCritic1 = loaded.trainedTargetCritic1;
        td3Agent.targetCritic2 = loaded.trainedTargetCritic2;

        if td3Agent.useGPU
            td3Agent.actor = moveNetworkToGPU(td3Agent.actor);
            td3Agent.critic1 = moveNetworkToGPU(td3Agent.critic1);
            td3Agent.critic2 = moveNetworkToGPU(td3Agent.critic2);
            td3Agent.targetActor = moveNetworkToGPU(td3Agent.targetActor);
            td3Agent.targetCritic1 = moveNetworkToGPU(td3Agent.targetCritic1);
            td3Agent.targetCritic2 = moveNetworkToGPU(td3Agent.targetCritic2);
        end

        fprintf('Pretrained TD3 networks loaded successfully\n');
    else
        % Initialize networks randomly
        fprintf('Initializing TD3 networks randomly...\n');
        fprintf('  State size: %d\n', stateSize);
        fprintf('  Action size: %d\n', actionSize);

        % Actor: state → action
        % Architecture: stateSize → 64 → 64 → 64 → actionSize
        td3Agent.actor = initializePaperActorNetwork(stateSize, actionSize);

        % Twin Critics: [state; action] → Q-value
        % Architecture: (stateSize + actionSize) → 64 → 64 → 32 → 32 → 16 → 1
        td3Agent.critic1 = initializePaperCriticNetwork(stateSize, actionSize);
        td3Agent.critic2 = initializePaperCriticNetwork(stateSize, actionSize);

        % Target networks (initialized as copies)
        td3Agent.targetActor = td3Agent.actor;
        td3Agent.targetCritic1 = td3Agent.critic1;
        td3Agent.targetCritic2 = td3Agent.critic2;

        % Move to GPU if available
        if td3Agent.useGPU
            td3Agent.actor = moveNetworkToGPU(td3Agent.actor);
            td3Agent.critic1 = moveNetworkToGPU(td3Agent.critic1);
            td3Agent.critic2 = moveNetworkToGPU(td3Agent.critic2);
            td3Agent.targetActor = moveNetworkToGPU(td3Agent.targetActor);
            td3Agent.targetCritic1 = moveNetworkToGPU(td3Agent.targetCritic1);
            td3Agent.targetCritic2 = moveNetworkToGPU(td3Agent.targetCritic2);
        end

        fprintf('TD3 Agent initialized with twin critics\n');
    end

    % State tracking
    td3Agent.lastState = [];
    td3Agent.lastAction = [];

    fprintf('\n=== TD3 Agent Configuration ===\n');
    fprintf('Actor LR: %.5f | Critic LR: %.5f\n', td3Agent.actorLR, td3Agent.criticLR);
    fprintf('Gamma: %.3f | Tau: %.4f\n', td3Agent.gamma, td3Agent.tau);
    fprintf('Policy Delay: %d | Target Noise: %.2f (clip ±%.2f)\n', ...
            td3Agent.policyDelay, td3Agent.targetNoiseStd, td3Agent.targetNoiseClip);
    fprintf('Batch Size: %d | Buffer Size: %d\n', td3Agent.batchSize, td3Agent.bufferSize);
    fprintf('==============================\n\n');
end

function net = moveNetworkToGPU(net)
    % Move all network weights to GPU
    weightFields = fieldnames(net);
    for i = 1:length(weightFields)
        field = weightFields{i};
        if isnumeric(net.(field)) && ~isa(net.(field), 'gpuArray')
            net.(field) = gpuArray(net.(field));
        end
    end
end
