function config = RLAMPSO_Config(mode)
    % Unified Configuration System for RLAM-PSO with All Improvements
    %
    % Provides centralized configuration for all system components with
    % backward compatibility for legacy code
    %
    % Usage:
    %   config = RLAMPSO_Config('default')    % Standard configuration
    %   config = RLAMPSO_Config('advanced')   % All improvements enabled
    %   config = RLAMPSO_Config('baseline')   % Original paper configuration
    %   config = RLAMPSO_Config('custom')     % Manual configuration
    %
    % Inputs:
    %   mode: Configuration preset (default: 'default')
    %
    % Outputs:
    %   config: Configuration structure

    if nargin < 1
        mode = 'baseline';
    end

    % ========== CORE CONFIGURATION ==========

    config = struct();
    config.mode = mode;
    config.version = '2.0';
    config.createdAt = datetime('now');

    % ===== IMPROVEMENT FLAGS =====

    % Improvement #2: Transformer Attention State
    config.useTransformerState = false;
    config.transformerHiddenDim = 64;
    config.transformerNumHeads = 4;
    config.transformerTemporalWindow = 5;  % Track last 5 states

    % Training Method Selection
    config.usePolicyGradient = false;   % Use Policy Gradient instead of DDPG/TD3

    % Improvement #3: TD3 with PER
    config.useTD3 = false;              % Use TD3 instead of DDPG
    config.usePER = false;              % Use Prioritized Experience Replay
    config.perAlpha = 0.6;              % Priority exponent
    config.perBetaStart = 0.4;          % Initial importance sampling weight
    config.perBetaFrames = 100000;      % Frames to anneal beta

    % Improvement #4: Curriculum Learning
    config.useCurriculum = false;
    config.curriculumSuccessThreshold = 0.8;  % 80% success to advance
    config.curriculumWindowSize = 20;         % Episodes to evaluate

    % Improvement #6: Transfer Learning with LoRA
    config.useLoRA = false;
    config.loraRank = 8;                % LoRA matrix rank
    config.loraPretrainedPath = '';     % Path to pretrained base weights
    config.loraFineTuneMode = false;    % Fine-tuning vs training from scratch

    % ===== DEEP LEARNING TOOLBOX =====
    % REQUIRED: MATLAB DL Toolbox (dlnetwork, automatic differentiation)
    % Will error during agent initialization if not available
    config.useDLToolbox = true;  % Always use DL Toolbox (required)

    % ===== PARALLEL TRAINING =====
    % DISABLED: dlnetwork objects don't serialize well for parfor
    % GPU-only sequential training is faster and more stable than parallel CPU training
    config.useParallel = false;         % Disabled due to dlnetwork serialization issues
    config.numParallelWorkers = 1;      % Single worker (GPU accelerated)
    config.numWorkers = 1;              % Alias for training functions compatibility
    config.parallelMode = 'episodes';   % 'episodes' or 'particles' (not used in GPU mode)

    % ===== NETWORK ARCHITECTURE =====

    config.stateSize = 15;              % 15 for sin-encoding, 64 for transformer
    config.actionSize = 20;             % 5 subgroups × 4 parameters
    config.numSubgroups = 5;

    % Actor architecture: state → 64 → 64 → 64 → action
    config.actorHiddenLayers = [64, 64, 64];

    % Critic architecture: [state; action] → 64 → 64 → 32 → 32 → 16 → 1
    config.criticHiddenLayers = [64, 64, 32, 32, 16];

    % ===== TRAINING HYPERPARAMETERS =====

    % Learning rates (paper-standard values for TD3/DDPG)
    config.actorLR = 1e-4;              % Standard TD3/DDPG actor learning rate
    config.criticLR = 1e-3;             % Standard TD3/DDPG critic learning rate

    % RL parameters
    config.gamma = 0.99;                % Discount factor
    config.tau = 0.001;                 % Soft update rate (CRITICAL: 0.125 caused training instability)

    % TD3-specific
    config.policyDelay = 2;             % Update actor every 2 critic updates
    config.targetNoiseStd = 0.2;        % Target policy smoothing noise
    config.targetNoiseClip = 0.5;       % Noise clip range

    % Exploration
    config.explorationNoiseStart = 0.1;
    config.explorationNoiseEnd = 0.01;
    config.explorationDecay = 0.995;

    % Experience replay (GPU-optimized)
    config.batchSize = 256;             % Balanced for RTX 3050 4GB VRAM (reduced from 2048)
    config.bufferSize = 1000000;        % 1M experiences

    % ===== TRAINING CONFIGURATION =====

    config.numEpisodes = 100;         % Total training episodes (aligned with official repo)
    config.maxIterations = 600;        % Max PSO iterations per episode (aligned with official repo)
    config.popSize = 40;               % PSO population size (aligned with official repo)

    % GPU settings (OPTIMIZED for single-GPU training)
    try
        config.useGPU = gpuDeviceCount > 0; % Auto-detect GPU
        if config.useGPU
            % GPU detected - configure for optimal performance
            gpu = gpuDevice;
            config.gpuName = gpu.Name;
            config.gpuMemory = gpu.AvailableMemory / 1e9;  % GB
            % Adaptive batch size for GPU training
            if config.gpuMemory > 8
                config.batchSize = 512;
            end
        end
    catch
        config.useGPU = false; % Parallel Computing Toolbox not available or no GPU
        config.gpuName = 'None';
        config.gpuMemory = 0;
        config.batchSize = 512;  % Smaller batches for CPU
    end

    % ===== GPU UTILIZATION OPTIMIZATION (2025 Research-Based) =====
    % Increase Update-To-Data (UTD) ratio for better GPU utilization
    % Based on recent research: optimal UTD ratio is 2-8 for TD3
    config.gradientStepsPerTrainingCall = 1;    % Multiple gradient steps per training call
    config.trainEveryNIterations = 1;          % Train during PSO (not just at end)
    config.enableMidEpisodeTraining = true;      % Enable training during episodes

    % Logging
    config.logInterval = 5;             % Print progress every N episodes (per worker)
    config.saveInterval = 100;          % Save checkpoint every N episodes
    config.savePath = 'models/rlampso_advanced.mat';
    config.verbose = false;             % Print detailed iteration logs (disable for training)

    % ===== CURRICULUM LEARNING =====

    config.useCurriculum = true;        % Enable curriculum learning
    config.curriculumSuccessThreshold = 0.8;  % 80% success rate to advance
    config.curriculumWindowSize = 20;   % Evaluate over 20 episodes

    % ===== REWARD SHAPING =====
    % Use paper-exact binary reward function (Equation 13)
    % Options: 'binary' (paper-exact), 'magnitude_aware' (alternative)
    config.rewardType = 'binary';  % Paper-exact binary reward (+1/-1)

    % CRITICAL FIX: Use episode-level rewards for proper credit assignment
    % When enabled, compares RLAM-PSO vs Baseline PSO on same environment
    % and gives episode reward = (baseline_fitness - rlam_fitness) / baseline_fitness
    % This solves credit assignment problem in stochastic environments
    config.useEpisodeLevelRewards = false;   % DISABLED - too slow, network not learning

    % ===== PSO CONFIGURATION =====

    config.numWaypoints = 5;
    config.mapSize = [400, 400, 100];

    % ===== MODE-SPECIFIC PRESETS =====
    % GPU-OPTIMIZED: All modes use single-GPU sequential training for stability

    switch lower(mode)
        case 'baseline'
            % Original paper configuration (DDPG only)
            % GPU-optimized sequential training
            config.useTransformerState = false;
            config.useTD3 = false;
            config.usePER = false;
            config.useCurriculum = false;
            config.useLoRA = false;
            config.useParallel = false;         % Disabled (GPU mode)
            config.numParallelWorkers = 1;      % Single GPU worker
            config.numWorkers = 1;              % Alias for compatibility
            config.numEpisodes = 100;           % UAV training episodes (Phase 2)
            config.maxIterations = 600;        % PSO iterations per episode (aligned with official repo)
            config.popSize = 40;               % PSO population size (aligned with official repo)
            % Batch size set by GPU detection above
            config.stateSize = 15;
            config.rewardType = 'binary';  % Paper-exact binary reward

        case 'advanced'
            % All improvements enabled (your full method)
            % GPU-optimized sequential training
            config.useTransformerState = true;
            config.useTD3 = true;
            config.usePER = true;
            config.useCurriculum = true;
            config.useLoRA = false;
            config.useParallel = false;         % Disabled (GPU mode)
            config.numParallelWorkers = 1;      % Single GPU worker
            config.numWorkers = 1;              % Alias for compatibility
            config.numEpisodes = 1000;         % UAV training episodes (aligned with official repo)
            config.maxIterations = 600;        % PSO iterations per episode (aligned with official repo)
            config.popSize = 40;               % PSO population size (aligned with official repo)
            % Batch size set by GPU detection above
            config.stateSize = 64;              % Transformer state size

        case 'td3_only'
            % TD3 + PER improvement only (ablation study)
            % GPU-optimized sequential training
            config.useTransformerState = false;
            config.useTD3 = true;
            config.usePER = true;
            config.useCurriculum = false;
            config.useLoRA = false;
            config.useParallel = false;         % Disabled (GPU mode)
            config.numParallelWorkers = 1;      % Single GPU worker
            config.numWorkers = 1;              % Alias for compatibility
            config.numEpisodes = 100;         % UAV training episodes (aligned with official repo)
            config.maxIterations = 600;        % PSO iterations per episode (aligned with official repo)
            config.popSize = 40;               % PSO population size (aligned with official repo)
            % Batch size set by GPU detection above
            config.stateSize = 15;

        case 'transformer_only'
            % Transformer state improvement only (ablation study)
            % GPU-optimized sequential training
            config.useTransformerState = true;
            config.useTD3 = false;
            config.usePER = false;
            config.useCurriculum = false;
            config.useLoRA = false;
            config.useParallel = false;         % Disabled (GPU mode)
            config.numParallelWorkers = 1;      % Single GPU worker
            config.numWorkers = 1;              % Alias for compatibility
            config.numEpisodes = 100;         % UAV training episodes (aligned with official repo)
            config.maxIterations = 600;        % PSO iterations per episode (aligned with official repo)
            config.popSize = 40;               % PSO population size (aligned with official repo)
            % Batch size set by GPU detection above
            config.stateSize = 64;              % Transformer state size

        case 'fast'
            % Fast testing mode with reduced parameters
            % GPU-optimized for quick experiments
            config.useTransformerState = false;
            config.useTD3 = true;
            config.usePER = false;              % Disabled for speed
            config.useCurriculum = false;
            config.useLoRA = false;
            config.useParallel = false;         % Disabled (GPU mode)
            config.numParallelWorkers = 1;      % Single GPU worker
            config.numWorkers = 1;              % Alias for compatibility
            config.numEpisodes = 5000;          % Half of production episodes for quick test
            config.maxIterations = 500;         % Half of production iterations
            config.popSize = 50;                % Half of production population
            % Batch size set by GPU detection above
            config.stateSize = 15;

        otherwise
            error(['Unknown mode: %s. Valid modes are:\n' ...
                   '  ''baseline''         - Original paper (DDPG)\n' ...
                   '  ''advanced''         - All improvements\n' ...
                   '  ''td3_only''         - Just TD3+PER\n' ...
                   '  ''transformer_only'' - Just Transformer\n' ...
                   '  ''fast''             - Quick testing'], mode);
    end

    % ===== PARAMETER MODE CONFIGURATION =====
    if ~isfield(config, 'paramMode')
        config.paramMode = '5subgroup';
    end

    switch config.paramMode
        case 'global'
            config.actionSize = 4;
            config.numSubgroups = 1;
        case '5subgroup'
            config.actionSize = 20;
            config.numSubgroups = 5;
        case 'per-particle'
            config.actionSize = config.popSize * 4;
            config.numSubgroups = config.popSize;
        otherwise
            error('Unknown paramMode: %s. Valid: global, 5subgroup, per-particle', config.paramMode);
    end

    % ===== VALIDATION =====

    % Transformer requires 64D state
    if config.useTransformerState
        config.stateSize = 64;
    end

    % PER requires TD3 or DDPG agent
    if config.usePER && ~config.useTD3
        fprintf('Note: PER enabled with DDPG (TD3 disabled)\n');
    end

    % LoRA requires pretrained weights path if in fine-tune mode
    if config.useLoRA && config.loraFineTuneMode && isempty(config.loraPretrainedPath)
        warning('LoRA fine-tuning enabled but no pretrained path specified!');
        fprintf('Set config.loraPretrainedPath before training.\n');
    end

    % ===== DISPLAY CONFIGURATION =====

    displayConfiguration(config);
end

function displayConfiguration(config)
    % Display configuration summary
    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║     RLAM-PSO Configuration (GPU-Optimized)            ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');

    fprintf('Mode: %s\n', config.mode);

    fprintf('\n--- Features Enabled ---\n');
    fprintf('  Transformer State: %s (state: %dD)\n', ...
            bool2str(config.useTransformerState), config.stateSize);
    fprintf('  TD3:               %s\n', bool2str(config.useTD3));
    fprintf('  PER:               %s\n', bool2str(config.usePER));
    fprintf('  Curriculum:        %s\n', bool2str(config.useCurriculum));

    fprintf('\n--- Compute Configuration ---\n');
    fprintf('  Mode:              GPU-Optimized Sequential\n');
    fprintf('  GPU:               %s', bool2str(config.useGPU));
    if config.useGPU && isfield(config, 'gpuName')
        fprintf(' (%s)\n', config.gpuName);
        fprintf('  GPU Memory:        %.1f GB available\n', config.gpuMemory);
    else
        fprintf(' (CPU fallback)\n');
    end
    fprintf('  Parallel:          %s (disabled for dlnetwork compatibility)\n', ...
            bool2str(config.useParallel));

    fprintf('\n--- Training Parameters ---\n');
    fprintf('  Episodes:          %d\n', config.numEpisodes);
    fprintf('  Population:        %d\n', config.popSize);
    fprintf('  Max Iterations:    %d\n', config.maxIterations);
    fprintf('  Batch Size:        %d (GPU-optimized)\n', config.batchSize);
    fprintf('  Actor LR:          %.5f\n', config.actorLR);
    fprintf('  Critic LR:         %.5f\n', config.criticLR);

    fprintf('\n--- GPU Optimization (2025) ---\n');
    fprintf('  Gradient Steps/Call:  %d\n', config.gradientStepsPerTrainingCall);
    fprintf('  Train Every:          %d iterations\n', config.trainEveryNIterations);
    fprintf('  Mid-Episode Training: %s\n', bool2str(config.enableMidEpisodeTraining));
    fprintf('  Estimated UTD Ratio:  ~%.1f\n', ...
            config.gradientStepsPerTrainingCall * (config.maxIterations / config.trainEveryNIterations));

    fprintf('\n════════════════════════════════════════════════════════\n\n');
end

function str = bool2str(value)
    % Convert boolean to string
    if value
        str = 'ON';
    else
        str = 'OFF';
    end
end
