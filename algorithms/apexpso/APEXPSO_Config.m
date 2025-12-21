function config = APEXPSO_Config(mode)
    % APEX-PSO Configuration: Advanced Parameter Exploration CrossQ-SAC for PSO
    %
    % State-of-the-art RL algorithm combining:
    %   - SAC (Soft Actor-Critic) for automatic entropy tuning
    %   - CrossQ optimizations (BatchNorm, no target networks, UTD=1)
    %   - Transformer state encoding with attention
    %   - Per-particle parameter adaptation (120D action space)
    %   - Multi-objective reward with curiosity bonus
    %
    % Based on 2024-2025 research:
    %   - CrossQ (ICLR 2024): Most sample-efficient with BatchNorm
    %   - SAC: Maximum entropy RL framework
    %   - Attention mechanisms for temporal context
    %
    % Usage:
    %   config = APEXPSO_Config()           % Default mode
    %   config = APEXPSO_Config('fast')     % Quick testing

    if nargin < 1
        mode = 'default';
    end

    % ========== CORE CONFIGURATION ==========

    config = struct();
    config.mode = mode;
    config.algorithm = 'APEX-PSO';
    config.version = '1.0';
    config.createdAt = datetime('now');

    % ===== ALGORITHM FEATURES =====

    % Core: SAC with CrossQ optimizations
    config.useSAC = true;                   % Use SAC instead of DDPG/TD3
    config.useAutomaticEntropyTuning = true; % SAC's automatic alpha tuning
    config.useBatchNorm = true;             % CrossQ: BatchNorm in all networks
    config.useTargetNetworks = false;       % CrossQ: No target networks (UTD=1)
    config.numCritics = 2;                  % Small ensemble (2 critics)

    % Advanced features
    config.useTransformerState = true;      % Attention-based state encoding
    config.usePerParticleActions = true;    % 120D: Individual particle params
    config.useMultiObjectiveReward = true;  % Fitness + diversity + curiosity
    config.useCuriosityBonus = true;        % Exploration bonus

    % ===== NETWORK ARCHITECTURE =====

    % State representation
    config.useAttention = true;
    config.attentionHeads = 4;              % Multi-head attention
    config.temporalWindow = 5;              % Track last 5 iterations
    config.stateSize = 45;                  % Rich state features

    % Action space: Per-particle parameters (40 particles × 3 params)
    config.actionSize = 120;                % 40 × [w, c1, c2]
    config.popSize = 40;                    % PSO population size
    config.paramsPerParticle = 3;           % [w, c1, c2]

    % Actor network: state → 128 → 128 → 128 → action
    % (Larger for 120D action space)
    config.actorHiddenLayers = [128, 128, 128];

    % Critic networks: [state; action] → 128 → 128 → 64 → 1
    % Two critics with BatchNorm (CrossQ approach)
    config.criticHiddenLayers = [128, 128, 64];

    % ===== SAC HYPERPARAMETERS =====

    % Learning rates (SAC standard)
    config.actorLR = 3e-4;                  % SAC standard
    config.criticLR = 3e-4;                 % SAC standard
    config.alphaLR = 3e-4;                  % Entropy coefficient LR

    % SAC-specific
    config.gamma = 0.99;                    % Discount factor
    config.tau = 0.005;                     % Soft update (only for optional targets)
    config.targetEntropy = -config.actionSize;  % Automatic: -dim(action)
    config.initAlpha = 0.2;                 % Initial entropy coefficient

    % CrossQ optimizations
    config.batchNormMomentum = 0.99;        % BatchNorm momentum
    config.batchNormEpsilon = 1e-5;         % BatchNorm epsilon
    config.utdRatio = 1;                    % Update-to-data ratio (CrossQ: 1)

    % ===== TRAINING CONFIGURATION =====
    %
    % OPTIMIZED FOR PUBLICATION-QUALITY RESULTS:
    % - 250 episodes ensures full convergence
    % - Buffer sized for 80% utilization (250*600 = 150k @ 150k capacity)
    % - Batch size balanced for GPU memory and learning stability

    config.numEpisodes = 250;               % Training episodes (matches run_comparison.m default)
    config.maxIterations = 600;             % PSO iterations per episode
    config.trainEveryNIterations = 1;       % Train every iteration
    config.gradientStepsPerTraining = 1;    % Gradient steps per training call

    % Experience replay (SAC uses large replay buffer)
    config.batchSize = 256;                 % Batch size (REDUCED for stability)
    config.bufferSize = 150000;             % 150k experiences (OPTIMIZED: 80% utilization @ 200 eps)
    config.warmupPeriod = 50;               % Random actions for first 50 iters

    % Exploration (SAC uses entropy, but add small action noise initially)
    config.explorationNoiseStart = 0.05;    % Small initial noise
    config.explorationNoiseEnd = 0.01;      % Even smaller final noise
    config.explorationDecay = 0.995;        % Decay per episode

    % ===== MULTI-OBJECTIVE REWARD =====

    % Reward components weights
    config.rewardWeightFitness = 1.0;       % Fitness improvement (primary)
    config.rewardWeightDiversity = 0.2;     % Maintain diversity
    config.rewardWeightConvergence = 0.1;   % Convergence speed bonus
    config.rewardWeightCuriosity = 0.05;    % Curiosity exploration bonus

    % Diversity tracking
    config.minDiversityThreshold = 0.01;    % Penalty if diversity too low
    config.diversityHistorySize = 10;       % Track diversity over 10 iters

    % ===== GPU CONFIGURATION =====

    try
        % Force disable GPU on macOS to avoid Parallel Computing Toolbox errors
        % (gpuArray support is limited/incompatible on newer macOS versions for this workload)
        config.useGPU = false; 
        
        % Legacy logic commented out:
        % config.useGPU = gpuDeviceCount > 0;
        % if config.useGPU
        %     gpu = gpuDevice;
        %     config.gpuName = gpu.Name;
        %     config.gpuMemory = gpu.AvailableMemory / 1e9;
        %     % Adaptive batch size
        %     if config.gpuMemory > 8
        %         config.batchSize = 1024;
        %     end
        % end
        
        config.gpuName = 'None';
        config.gpuMemory = 0;
        config.batchSize = 256;
    catch
        config.useGPU = false;
        config.gpuName = 'None';
        config.gpuMemory = 0;
        config.batchSize = 256;  % Smaller for CPU
    end

    % ===== PSO CONFIGURATION =====

    config.numWaypoints = 5;
    config.mapSize = [400, 400, 100];

    % ===== LOGGING =====

    config.logInterval = 5;                 % Log every 5 episodes
    config.saveInterval = 50;               % Save every 50 episodes
    config.savePath = 'models/apexpso.mat';
    config.verbose = false;                 % Detailed logs

    % ===== PRETRAINED MODEL =====

    config.pretrainedModelPath = '';        % Path to pretrained model (empty = train from scratch)

    % ===== MODE-SPECIFIC PRESETS =====

    switch lower(mode)
        case 'fast'
            % Quick testing mode (for debugging)
            config.numEpisodes = 50;
            config.maxIterations = 300;
            config.batchSize = 128;
            config.bufferSize = 20000;          % Right-sized: 50*300 = 15k @ 75% full
            config.warmupPeriod = 20;

        case 'ablation_no_attention'
            % Ablation: No attention/transformer
            config.useTransformerState = false;
            config.stateSize = 15;

        case 'ablation_subgroup'
            % Ablation: Use 5-subgroup instead of per-particle
            config.usePerParticleActions = false;
            config.actionSize = 20;  % 5 × 4 params

        case 'ablation_simple_reward'
            % Ablation: Binary reward only
            config.useMultiObjectiveReward = false;
            config.useCuriosityBonus = false;

        case 'ablation_no_crossq'
            % Ablation: Standard SAC without CrossQ optimizations
            config.useBatchNorm = false;        % No BatchNorm (use standard networks)
            config.useTargetNetworks = true;    % Use target networks (standard SAC)

        case 'online'
            % Online learning mode: No pretraining, learn during actual PSO run
            config.numEpisodes = 1;              % Single episode = actual problem
            % maxIterations will be set by caller (from defaults.maxIterations)
            % If not set externally, default to 600
            if ~isfield(config, 'maxIterations')
                config.maxIterations = 600;      % Default for production runs
            end
            config.warmupPeriod = min(50, floor(config.maxIterations * 0.1));  % 10% of iterations or 50
            config.trainEveryNIterations = 1;    % Train every iteration
            config.batchSize = 256;              % Smaller batch for online
            config.bufferSize = 10000;           % Smaller buffer (only 600 samples)

        case 'pretrained'
            % Use pretrained model for optimization (no additional training)
            config.numEpisodes = 1;              % Single optimization run
            config.maxIterations = 600;          % Same as other algorithms
            config.pretrainedModelPath = 'models/apexpso.mat';  % Load trained model
            config.warmupPeriod = 0;             % No warmup needed
            config.trainEveryNIterations = 999999;  % Disable training (just use policy)
            config.batchSize = 256;
            config.bufferSize = 10000;

        otherwise
            % Default mode: All features enabled
    end

    % ===== PARAMETER MODE CONFIGURATION =====
    if isfield(config, 'paramMode')
        switch config.paramMode
            case 'global'
                config.usePerParticleActions = false;
                config.actionSize = 3;
            case '5subgroup'
                config.usePerParticleActions = false;
                config.actionSize = 15;
            case 'per-particle'
                config.usePerParticleActions = true;
                config.actionSize = config.popSize * config.paramsPerParticle;
        end
    end

    % ===== VALIDATION =====

    % Ensure consistency
    if config.usePerParticleActions
        assert(config.actionSize == config.popSize * config.paramsPerParticle, ...
            'Action size mismatch for per-particle params');
    end

    % ===== DISPLAY CONFIGURATION =====

    displayConfiguration(config);
end

function displayConfiguration(config)
    % Display configuration summary
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║     APEX-PSO Configuration (SAC-CrossQ)                 ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    fprintf('Mode: %s\n', config.mode);
    fprintf('Algorithm: %s v%s\n', config.algorithm, config.version);

    fprintf('\n--- Core Features ---\n');
    fprintf('  SAC Framework:        %s (automatic entropy tuning)\n', bool2str(config.useSAC));
    fprintf('  CrossQ BatchNorm:     %s\n', bool2str(config.useBatchNorm));
    fprintf('  Target Networks:      %s (CrossQ approach)\n', bool2str(config.useTargetNetworks));
    fprintf('  Critic Ensemble:      %d critics\n', config.numCritics);
    fprintf('  Transformer State:    %s (%dD, %d heads, window=%d)\n', ...
        bool2str(config.useTransformerState), config.stateSize, ...
        config.attentionHeads, config.temporalWindow);
    fprintf('  Per-Particle Actions: %s (%dD)\n', ...
        bool2str(config.usePerParticleActions), config.actionSize);
    fprintf('  Multi-Obj Reward:     %s (fitness+diversity+curiosity)\n', ...
        bool2str(config.useMultiObjectiveReward));

    fprintf('\n--- Training Parameters ---\n');
    fprintf('  Episodes:             %d\n', config.numEpisodes);
    fprintf('  Population:           %d particles\n', config.popSize);
    fprintf('  Max Iterations:       %d\n', config.maxIterations);
    fprintf('  Batch Size:           %d\n', config.batchSize);
    fprintf('  Learning Rates:       Actor=%.5f, Critic=%.5f, Alpha=%.5f\n', ...
        config.actorLR, config.criticLR, config.alphaLR);
    fprintf('  UTD Ratio:            %d (CrossQ optimized)\n', config.utdRatio);
    fprintf('  Train Frequency:      Every %d iterations\n', config.trainEveryNIterations);

    fprintf('\n--- Compute ---\n');
    fprintf('  GPU:                  %s', bool2str(config.useGPU));
    if config.useGPU
        fprintf(' (%s, %.1f GB)\n', config.gpuName, config.gpuMemory);
    else
        fprintf(' (CPU mode)\n');
    end

    fprintf('\n══════════════════════════════════════════════════════════\n\n');
end

function str = bool2str(value)
    if value
        str = 'ON';
    else
        str = 'OFF';
    end
end
