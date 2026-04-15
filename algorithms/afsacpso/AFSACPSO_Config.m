function config = AFSACPSO_Config(mode)
    % AFSACPSO configuration with plain SAC critics plus attractor-field control.
    %
    % RL-PSO algorithm combining:
    %   - SAC (Soft Actor-Critic) with twin critics
    %   - Target networks and automatic entropy tuning
    %   - Rank-residual parameter adaptation (9D latent action)
    %   - Simple fitness-improvement reward
    %
    % Usage:
    %   config = AFSACPSO_Config()           % Default mode
    %   config = AFSACPSO_Config('fast')     % Quick testing

    if nargin < 1
        mode = 'online';
    end

    % ========== CORE CONFIGURATION ==========

    config = struct();
    config.mode = mode;
    config.algorithm = 'AFSACPSO';
    config.version = '4.8-sac-attractorfield';


    % ===== ALGORITHM FEATURES =====

    % Core: SAC with configurable critic-side research branches
    config.useSAC = true;                   % Use SAC instead of DDPG/TD3
    config.useAutomaticEntropyTuning = true; % SAC's automatic alpha tuning
    config.useCrossQCritic = false;         % Retired after failing LOO significance
    config.useREDQCritic = false;           % Research branch: critic ensemble + random subset targets
    config.useSimBaBackbone = false;        % Disabled in retained stack
    config.useAQECritic = false;            % Retained as an optional research branch
    config.useDroQCritic = false;           % Retired after failing LOO significance
    config.useObservationNormalization = false;
    config.observationNormClip = 5.0;
    config.useBatchNorm = false;            % Legacy alias kept for ablations only
    config.useCriticBatchNorm = false;
    config.useActorBatchNorm = false;
    config.useJointCriticBatchForBN = false;
    config.useWeightNormCritic = false;
    config.criticWeightNormRadius = 1.0;
    config.useTargetNetworks = true;
    config.targetCriticTrainMode = false;
    config.numCritics = 2;                  % Standard twin critics; REDQ/AQE reinterpret this
    config.redqNumCritics = 5;
    config.redqTargetSubsetSize = 2;
    config.redqTargetMode = 'min';
    config.redqPolicyUpdateDelay = 5;
    config.redqWarmupSteps = 128;

    % Advanced features
    config.useCrossScaleState = false;      % Cross-scale encoder retired from retained stack
    config.usePerParticleActions = false;   % Use low-dim latent control by default
    config.useAttractorFieldControl = true;   % Expand latent action to per-particle params
    config.usePrioritizedReplay = false;
    config.useResidualCriticDecomposition = false;
    config.usePilarReturns = false;
    config.pilarEffectiveNStep = 3;
    config.pilarLongHorizon = 6;
    config.pilarMixCoefficient = 0.406;
    config.useTQCCritic = false;
    config.useD2RLBackbone = false;
    config.useEmphasizingRecentExperience = false;
    config.ereEtaStart = 0.996;
    config.ereEtaEnd = 1.0;
    config.ereExponentScale = 1000;
    config.ereAnnealSteps = 1200;
    config.ereMinRecentSize = 128;
    config.useDelayedPolicyUpdates = false;
    config.actorUpdateInterval = 1;

    % ===== NETWORK ARCHITECTURE =====

    % State representation
    config.useAttention = false;
    config.attentionHeads = 4;              % Multi-head attention
    config.temporalWindow = 8;              % Track longer temporal context
    config.stateSize = 15;                  % Flat temporal summary state
    config.crossScaleBlockSize = 9;
    config.crossScaleBlockNames = {'current', 'short_ema', 'long_ema', 'volatility', 'trend'};
    config.useCrossScaleBranching = false;
    config.useCrossScaleGatedFusion = false;
    config.actorCrossScaleBranchWidths = [32, 28, 28, 16, 16];
    config.criticCrossScaleBranchWidths = [28, 28, 28, 16, 16];
    config.crossScaleActorGateHiddenWidth = 64;
    config.crossScaleCriticGateHiddenWidth = 64;
    config.crossScalePredictiveGain = 0.25;
    config.crossScaleTrendGain = 0.20;
    config.crossScaleAccelerationGain = 0.02;
    config.crossScaleConsistencyMix = 0.50;
    config.crossScaleQuantizationSharpness = 20;
    config.crossScaleQuantizationCenters = [0.15, 0.38, 0.62, 0.85];

    % Action space: low-dimensional attractor-field control
    config.paramMode = 'attractor-field';   % Default mode; overridden per-mode below
    config.actionSize = 9;                  % [w,c1,c2] residual coefficients
    config.popSize = 100;                   % Benchmark-aligned default population size
    config.paramsPerParticle = 3;           % [w, c1, c2]

    % Flat MLP actor remains the non-component baseline policy network.
    config.actorHiddenLayers = [256, 256];
    config.simbaActorWidth = 128;
    config.simbaActorBlocks = 3;

    config.criticHiddenLayers = [256, 256, 128];
    config.aqeHeadsPerCritic = 3;
    config.aqeKeepHeads = 2;
    config.simbaCriticWidth = 192;
    config.simbaCriticBlocks = 3;
    config.droqCriticWidth = 256;
    config.droqCriticDepth = 3;
    config.droqDropoutProbability = 0.01;
    config.tqcNumQuantiles = 25;
    config.tqcDropQuantilesPerCritic = 2;
    config.tqcHuberKappa = 1.0;
    config.criticActionBranchWidth = 64;
    config.criticAdvantageHiddenWidth = 128;

    % ===== SAC HYPERPARAMETERS =====

    % Learning rates (SAC standard)
    config.actorLR = 3e-4;                  % SAC standard
    config.criticLR = 3e-4;                 % SAC standard
    config.alphaLR = 3e-4;                  % Entropy coefficient LR
    config.actorAdamBeta1 = 0.9;
    config.actorAdamBeta2 = 0.999;
    config.criticAdamBeta1 = 0.9;
    config.criticAdamBeta2 = 0.999;

    % SAC-specific
    config.gamma = 0.99;                    % Discount factor
    config.tau = 0.005;                     % Soft update (only for optional targets)
    config.targetEntropy = -config.actionSize;  % Automatic: -dim(action)
    config.initAlpha = 0.2;                 % Initial entropy coefficient
    config.entropyAnnealStrength = 0.40;    % Reduce exploration late in run
    config.entropyAnnealSteps = 1200;       % Anneal horizon (train steps)

    % Legacy normalization knobs kept for ablations and backwards compatibility
    config.batchNormMomentum = 0.99;        % BatchNorm momentum
    config.batchNormEpsilon = 1e-5;         % BatchNorm epsilon
    config.utdRatio = 1;                    % CrossQ reaches strong sample efficiency at low UTD

    % ===== TRAINING CONFIGURATION =====
    %
    % Benchmark-aligned defaults for the retained controller.

    config.numEpisodes = 250;               % Training episodes (matches run_comparison.m default)
    config.maxIterations = 1000;            % PSO iterations per episode
    config.trainEveryNIterations = 1;       % Train every iteration
    config.gradientStepsPerTraining = config.utdRatio;  % Keep UTD explicit in training loop

    % Experience replay (SAC uses large replay buffer)
    config.batchSize = 256;                 % Batch size for SAC training
    config.bufferSize = 150000;             % 150k experiences
    config.warmupPeriod = 0;                % No warmup (train immediately)

    % Exploration (SAC uses entropy, but add small action noise initially)
    config.explorationNoiseStart = 0.05;
    config.explorationNoiseEnd = 0.01;
    config.explorationDecay = 0.995;

    % ===== GPU CONFIGURATION =====
    % Force-disabled on macOS (incompatible with this workload)
    config.useGPU = false;
    config.gpuName = 'None';
    config.gpuMemory = 0;

    % ===== PSO CONFIGURATION =====

    config.numWaypoints = 5;
    config.mapSize = [400, 400, 100];
    % Post-PSO waypoint-decomposed Nelder-Mead local search (production default)
    config.useTrajectoryPolish = true;
    config.polishNumStarts = 30;
    config.polishNumSeedPaths = 3;
    config.polishMaxPasses = 8;
    config.polishEvalsPerWaypoint = 400;
    config.polishStepXY = 100;
    config.polishStepZ = 15;

    % Velocity clamping: per-dimension (aligned with SAC-SAPSO paper)
    config.velocityClampMode = 'per-dimension';
    config.velocityClampDelta = 0.5;

    % Observation interval: act every nt iterations
    % SAC-SAPSO paper: nt=125 @ 5000 iters → 40 transitions/episode.
    % Scale proportionally: nt = maxIterations / 40.
    config.observationInterval = max(1, floor(config.maxIterations / 40));

    % ===== LOGGING =====

    config.logInterval = 5;                 % Log every 5 episodes
    config.saveInterval = 50;               % Save every 50 episodes
    config.savePath = 'models/afsacpso.mat';
    config.verbose = false;                 % Detailed logs
    config.disableVisualization = false;    % Disable plotting in batch research runs

    % ===== MODE-SPECIFIC PRESETS =====

    switch lower(mode)
        case 'fast'
            % Quick testing mode (for debugging)
            config.numEpisodes = 50;
            config.maxIterations = 300;
            config.batchSize = 64;
            config.bufferSize = 20000;          % Right-sized: 50*300 = 15k @ 75% full
            config.warmupPeriod = 0;

        case 'pretrain'
            % Pre-training mode: train SAC across many episodes on
            % benchmark functions, then deploy frozen policy.
            %
            % PSO settings MUST match deployment so the policy sees the
            % same state distributions at inference: popSize=100, maxIter=1000.
            %
            % SAC hyperparameters follow von Eschwege & Engelbrecht 2024:
            %   gamma=1, tau=0.005, [256,256] networks, buffer=10^6.
            % With nt=iter/40, each episode yields 40 transitions.
            % 5000 episodes × 40 = 200k gradient steps (matching paper's 2×10^5).
            config.numEpisodes = 5000;
            config.maxIterations = 1000;         % Aligned with deployment
            config.popSize = 100;                % Aligned with deployment
            config.warmupPeriod = 0;
            config.trainEveryNIterations = 1;
            config.batchSize = 256;              % Paper: batch 256
            config.utdRatio = 1;                 % Paper: 1 gradient step per transition
            config.gradientStepsPerTraining = 1;
            config.bufferSize = 1000000;         % Paper: 10^6
            config.paramMode = 'attractor-field';

            % Paper hyperparameters
            config.actorHiddenLayers = [256, 256];   % Paper: actor 256
            config.criticHiddenLayers = [256, 256];  % Paper: critic 256
            config.gamma = 1.0;                  % Paper: gamma = 1
            config.actorLR = 3e-4;               % SAC standard
            config.criticLR = 3e-4;
            config.alphaLR = 3e-4;
            config.tau = 0.005;                  % Paper: tau = 0.005

            % No local search during pre-training
            config.useTrajectoryPolish = false;

            % Checkpoint saving
            config.savePath = fullfile('models', 'AFSACPSO', 'pretrained_sac.mat');
            config.saveInterval = 25;            % Checkpoint every 25 episodes

        case 'sacpso-paper'
            % Paper-aligned SAC-SAPSO mode (von Eschwege & Engelbrecht 2024).
            % Matches the paper's algorithm exactly:
            %   - State: popSize+3 (per-particle velocity + stability + infeasible + completion)
            %   - Action: 3D global (w, c1, c2) shared by all particles
            %   - Observation interval: nt (act every nt iterations)
            %   - Reward: paper's piecewise relative reward (Eq. 25)
            %   - Velocity clamping: per-dimension
            %   - No PSO restarts, no local search
            % With nt=iter/40, each episode yields 40 transitions.
            % 5000 episodes × 40 = 200k gradient steps (matching paper's 2×10^5).
            config.numEpisodes = 5000;
            config.maxIterations = 1000;         % Aligned with UAV deployment
            config.popSize = 100;                % Aligned with UAV deployment
            config.warmupPeriod = 0;
            config.trainEveryNIterations = 1;    % But only stores/trains at nt intervals

            % Paper's SAC hyperparameters (Table 2)
            config.gamma = 1.0;
            config.tau = 0.005;
            config.actorLR = 3e-4;
            config.criticLR = 3e-4;
            config.alphaLR = 3e-4;
            config.actorHiddenLayers = [256, 256];
            config.criticHiddenLayers = [256, 256];
            config.batchSize = 256;
            config.bufferSize = 1000000;
            config.utdRatio = 1;
            config.gradientStepsPerTraining = 1;

            % Paper algorithm design
            config.paramMode = 'global';         % 3D action: one (w,c1,c2) for all particles
            config.actionSize = 3;
            config.stateMode = 'paper';          % Use buildPaperState instead of extractRichFeatures
            config.stateSize = config.popSize + 3;  % ns + 3
            config.rewardMode = 'paper';         % Use calculatePaperReward (Eq. 25)
            config.useAttractorFieldControl = false;
            config.usePerParticleActions = false;
            config.useTrajectoryPolish = false;
            config.temporalWindow = 1;           % No temporal encoding needed

            config.savePath = fullfile('models', 'AFSACPSO', 'pretrained_sacpso_paper.mat');
            config.saveInterval = 25;

        otherwise
            % Default mode: All features enabled
    end

    % ===== PARAMETER MODE CONFIGURATION =====
    if isfield(config, 'paramMode')
        config = applyParamMode(config);
    end

    % Keep the training loop aligned with the configured UTD ratio unless
    % a caller explicitly overrides gradientStepsPerTraining later.
    if ~(isfield(config, 'gradientStepsPerTraining') && ~isempty(config.gradientStepsPerTraining))
        config.gradientStepsPerTraining = max(1, round(config.utdRatio));
    else
        config.gradientStepsPerTraining = max(1, round(config.gradientStepsPerTraining));
    end
    config.ereMinRecentSize = max(1, min(config.bufferSize, round(config.ereMinRecentSize)));
    config.ereAnnealSteps = max(1, round(config.ereAnnealSteps));

    % ===== VALIDATION =====

    % Ensure consistency
    if config.usePerParticleActions
        assert(config.actionSize == config.popSize * config.paramsPerParticle, ...
            'Action size mismatch for per-particle params');
    end
    if config.useREDQCritic
        config.useCrossQCritic = false;
        config.useAQECritic = false;
        config.useDroQCritic = false;
        config.useTQCCritic = false;
        config.useCriticBatchNorm = false;
        config.useActorBatchNorm = false;
        config.useJointCriticBatchForBN = false;
        config.useWeightNormCritic = false;
        config.useTargetNetworks = true;
        config.redqNumCritics = max(3, round(max(config.numCritics, config.redqNumCritics)));
        config.numCritics = config.redqNumCritics;
        config.redqTargetSubsetSize = max(2, min(config.numCritics, round(config.redqTargetSubsetSize)));
        config.utdRatio = max(1, round(config.utdRatio));
        config.gradientStepsPerTraining = max(1, round(config.gradientStepsPerTraining));
        config.redqPolicyUpdateDelay = max(1, round(config.redqPolicyUpdateDelay));
        config.useDelayedPolicyUpdates = config.redqPolicyUpdateDelay > 1;
        if config.useDelayedPolicyUpdates
            config.actorUpdateInterval = config.redqPolicyUpdateDelay;
        else
            config.actorUpdateInterval = 1;
        end
    end
    if config.useCrossQCritic
        config.useREDQCritic = false;
        config.useAQECritic = false;
        config.useDroQCritic = false;
        config.useTQCCritic = false;
        config.useCriticBatchNorm = true;
        config.useJointCriticBatchForBN = true;
        config.useTargetNetworks = false;
        config.utdRatio = 1;
        config.gradientStepsPerTraining = 1;
    end
    if config.useTQCCritic
        config.useCrossQCritic = false;
        config.useREDQCritic = false;
        config.useAQECritic = false;
        config.useDroQCritic = false;
        config.useCriticBatchNorm = false;
        config.useActorBatchNorm = false;
        config.useJointCriticBatchForBN = false;
        config.useWeightNormCritic = false;
        config.useTargetNetworks = true;
        config.numCritics = 2;
        config.utdRatio = 1;
        config.gradientStepsPerTraining = 1;
        config.useDelayedPolicyUpdates = false;
        config.actorUpdateInterval = 1;
        config.tqcNumQuantiles = max(5, round(config.tqcNumQuantiles));
        config.tqcDropQuantilesPerCritic = max(0, min(config.tqcNumQuantiles - 1, ...
            round(config.tqcDropQuantilesPerCritic)));
        config.tqcHuberKappa = max(eps, config.tqcHuberKappa);
    end
    if config.useAQECritic
        totalAqeHeads = config.numCritics * config.aqeHeadsPerCritic;
        assert(totalAqeHeads >= 2, 'AQE requires at least two total critic heads.');
        config.aqeKeepHeads = max(1, min(totalAqeHeads, round(config.aqeKeepHeads)));
    end

    % ===== DISPLAY CONFIGURATION =====
    quietFlag = strcmpi(strtrim(getenv('APEXPSO_CONFIG_QUIET')), '1');
    if ~quietFlag && ~(isfield(config, 'suppressConfigDisplay') && config.suppressConfigDisplay)
        displayConfiguration(config);
    end
end

function displayConfiguration(config)
    % Display configuration summary
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║ AFSACPSO Configuration (Retained SAC + Attractor-Field) ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    fprintf('Mode: %s\n', config.mode);
    fprintf('Algorithm: %s v%s\n', config.algorithm, config.version);

    fprintf('\n--- Core Features ---\n');
    fprintf('  SAC Framework:        %s (automatic entropy tuning)\n', bool2str(config.useSAC));
    fprintf('  CrossQ Critic:        %s\n', bool2str(config.useCrossQCritic));
    fprintf('  REDQ Critic:          %s (N=%d, M=%d, mode=%s, delay=%d)\n', ...
        bool2str(config.useREDQCritic), config.numCritics, ...
        config.redqTargetSubsetSize, config.redqTargetMode, config.actorUpdateInterval);
    fprintf('  AQE Critic:           %s (%d heads/critic, keep=%d)\n', ...
        bool2str(config.useAQECritic), config.aqeHeadsPerCritic, config.aqeKeepHeads);
    fprintf('  DroQ Critic:          %s\n', bool2str(config.useDroQCritic));
    fprintf('  SimBa Backbone:       %s\n', bool2str(config.useSimBaBackbone));
    fprintf('  PiLaR Returns:        %s (eff=%d, long=%d, c=%.3f)\n', ...
        bool2str(config.usePilarReturns), config.pilarEffectiveNStep, ...
        config.pilarLongHorizon, config.pilarMixCoefficient);
    fprintf('  TQC Critic:           %s (%d quantiles, drop=%d/critic)\n', ...
        bool2str(config.useTQCCritic), config.tqcNumQuantiles, config.tqcDropQuantilesPerCritic);
    fprintf('  D2RL Backbone:        %s\n', bool2str(config.useD2RLBackbone));
    fprintf('  ERE Replay:           %s (eta %.3f -> %.3f, cmin=%d)\n', ...
        bool2str(config.useEmphasizingRecentExperience), config.ereEtaStart, ...
        config.ereEtaEnd, config.ereMinRecentSize);
    fprintf('  Delayed Actor:        %s (interval=%d)\n', ...
        bool2str(config.useDelayedPolicyUpdates), config.actorUpdateInterval);
    fprintf('  Obs Normalization:    %s (clip=%.1f)\n', ...
        bool2str(config.useObservationNormalization), config.observationNormClip);
    fprintf('  Critic BatchNorm:     %s (legacy ablation)\n', bool2str(config.useCriticBatchNorm));
    fprintf('  Target Networks:      %s (train-mode targets)\n', bool2str(config.useTargetNetworks));
    if config.useREDQCritic
        fprintf('  Critic Ensemble:      %d critics (random target subset=%d)\n', ...
            config.numCritics, config.redqTargetSubsetSize);
    elseif config.useAQECritic
        fprintf('  Critic Ensemble:      %d critics (%d total AQE heads)\n', ...
            config.numCritics, config.numCritics * max(1, config.aqeHeadsPerCritic));
    else
        fprintf('  Critic Ensemble:      %d critics\n', config.numCritics);
    end
    fprintf('  Cross-Scale State:    %s (%dD, window=%d)\n', ...
        bool2str(config.useCrossScaleState), config.stateSize, config.temporalWindow);
    fprintf('  Attractor-Field Ctrl: %s (%dD latent)\n', ...
        bool2str(config.useRankResidualControl), config.actionSize);
    fprintf('  Per-Particle Actions: %s\n', bool2str(config.usePerParticleActions));
    fprintf('  Reward Signal:        Relative fitness-improvement reward\n');

    fprintf('\n--- Training Parameters ---\n');
    fprintf('  Episodes:             %d\n', config.numEpisodes);
    fprintf('  Population:           %d particles\n', config.popSize);
    fprintf('  Max Iterations:       %d\n', config.maxIterations);
    fprintf('  Batch Size:           %d\n', config.batchSize);
    fprintf('  Learning Rates:       Actor=%.5f, Critic=%.5f, Alpha=%.5f\n', ...
        config.actorLR, config.criticLR, config.alphaLR);
    fprintf('  Adam Betas:           Actor=(%.2f, %.3f), Critic=(%.2f, %.3f)\n', ...
        config.actorAdamBeta1, config.actorAdamBeta2, ...
        config.criticAdamBeta1, config.criticAdamBeta2);
    fprintf('  UTD Ratio:            %d\n', config.utdRatio);
    fprintf('  Train Frequency:      Every %d iterations\n', config.trainEveryNIterations);
    fprintf('  Actor Backbone:       width=%d, blocks=%d\n', config.simbaActorWidth, config.simbaActorBlocks);
    fprintf('  Critic Backbone:      width=%d, blocks=%d\n', config.simbaCriticWidth, config.simbaCriticBlocks);

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
