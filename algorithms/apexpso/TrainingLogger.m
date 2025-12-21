classdef TrainingLogger < handle
    % TrainingLogger - Tracks and saves training metrics for APEXPSO
    %
    % This logger captures detailed training metrics to compare:
    %   - Standard SAC (with target networks, no BatchNorm)
    %   - CrossQ-SAC (no target networks, with BatchNorm)
    %
    % Metrics tracked per episode:
    %   - Episode fitness (best path found)
    %   - Episode reward (cumulative RL reward)
    %   - Actor loss, Critic losses (both critics)
    %   - Alpha (entropy coefficient)
    %   - Mean Q-values, Policy entropy
    %   - Gradient norms, Training time
    %   - Sample efficiency metrics
    %
    % Usage:
    %   logger = TrainingLogger(config);
    %   logger.startEpisode(episode);
    %   logger.logIteration(reward, fitness, losses, qValues);
    %   logger.endEpisode();
    %   logger.save('models/training_log.mat');
    %   logger.plotTrainingCurves();

    properties
        config              % Configuration structure
        modelName           % Name of model being trained
        startTime           % Training start time

        % Per-episode metrics
        episodeRewards      % Cumulative reward per episode
        episodeFitness      % Best fitness per episode
        episodeDuration     % Time per episode (seconds)

        % Per-episode loss tracking
        actorLoss           % Mean actor loss per episode
        critic1Loss         % Mean critic 1 loss per episode
        critic2Loss         % Mean critic 2 loss per episode
        alphaValue          % Mean alpha value per episode

        % Per-episode Q-value and entropy tracking
        meanQValue          % Mean Q-value estimates
        maxQValue           % Max Q-value estimates
        policyEntropy       % Mean policy entropy

        % Gradient norm tracking (for stability analysis)
        actorGradNorm       % Actor gradient L2 norm
        criticGradNorm      % Critic gradient L2 norm

        % Sample efficiency
        totalSamples        % Cumulative environment samples
        rewardPerSample     % Sample efficiency metric

        % Current episode buffers
        currentEpisode      % Current episode number
        episodeIterations   % Iterations in current episode
        episodeRewardBuffer % Reward accumulator for current episode
        episodeLossBuffer   % Loss values for current episode
    end

    methods
        function obj = TrainingLogger(config)
            % Initialize logger with configuration
            obj.config = config;
            obj.modelName = config.mode;
            obj.startTime = datetime('now');

            % Pre-allocate arrays
            numEpisodes = config.numEpisodes;
            obj.episodeRewards = zeros(1, numEpisodes);
            obj.episodeFitness = zeros(1, numEpisodes);
            obj.episodeDuration = zeros(1, numEpisodes);

            obj.actorLoss = zeros(1, numEpisodes);
            obj.critic1Loss = zeros(1, numEpisodes);
            obj.critic2Loss = zeros(1, numEpisodes);
            obj.alphaValue = zeros(1, numEpisodes);

            obj.meanQValue = zeros(1, numEpisodes);
            obj.maxQValue = zeros(1, numEpisodes);
            obj.policyEntropy = zeros(1, numEpisodes);

            obj.actorGradNorm = zeros(1, numEpisodes);
            obj.criticGradNorm = zeros(1, numEpisodes);

            obj.totalSamples = zeros(1, numEpisodes);
            obj.rewardPerSample = zeros(1, numEpisodes);

            obj.currentEpisode = 0;
            obj.episodeIterations = 0;

            fprintf('📊 Training logger initialized: %s\n', obj.modelName);
            fprintf('   Episodes: %d | Max Iterations: %d\n', ...
                numEpisodes, config.maxIterations);
        end

        function startEpisode(obj, episode)
            % Start tracking a new episode
            obj.currentEpisode = episode;
            obj.episodeIterations = 0;
            obj.episodeRewardBuffer = 0;
            obj.episodeLossBuffer = struct();
            obj.episodeLossBuffer.actor = [];
            obj.episodeLossBuffer.critic1 = [];
            obj.episodeLossBuffer.critic2 = [];
            obj.episodeLossBuffer.alpha = [];
            obj.episodeLossBuffer.qValues = [];
            obj.episodeLossBuffer.entropy = [];
            obj.episodeLossBuffer.actorGrad = [];
            obj.episodeLossBuffer.criticGrad = [];
        end

        function logIteration(obj, reward, fitness, losses)
            % Log metrics for a single PSO iteration
            %
            % Inputs:
            %   reward: RL reward for this iteration
            %   fitness: Current best fitness
            %   losses: struct with fields:
            %     - actorLoss, critic1Loss, critic2Loss
            %     - alpha, qValue, entropy
            %     - actorGradNorm, criticGradNorm (optional)

            obj.episodeIterations = obj.episodeIterations + 1;
            obj.episodeRewardBuffer = obj.episodeRewardBuffer + reward;

            % Store losses for averaging at episode end
            if ~isempty(losses)
                if isfield(losses, 'actorLoss')
                    obj.episodeLossBuffer.actor(end+1) = losses.actorLoss;
                end
                if isfield(losses, 'critic1Loss')
                    obj.episodeLossBuffer.critic1(end+1) = losses.critic1Loss;
                end
                if isfield(losses, 'critic2Loss')
                    obj.episodeLossBuffer.critic2(end+1) = losses.critic2Loss;
                end
                if isfield(losses, 'alpha')
                    obj.episodeLossBuffer.alpha(end+1) = losses.alpha;
                end
                if isfield(losses, 'qValue')
                    obj.episodeLossBuffer.qValues(end+1) = losses.qValue;
                end
                if isfield(losses, 'entropy')
                    obj.episodeLossBuffer.entropy(end+1) = losses.entropy;
                end
                if isfield(losses, 'actorGradNorm')
                    obj.episodeLossBuffer.actorGrad(end+1) = losses.actorGradNorm;
                end
                if isfield(losses, 'criticGradNorm')
                    obj.episodeLossBuffer.criticGrad(end+1) = losses.criticGradNorm;
                end
            end
        end

        function endEpisode(obj, finalFitness, episodeTime)
            % Finalize episode logging and compute statistics
            ep = obj.currentEpisode;

            % Store episode-level metrics
            obj.episodeRewards(ep) = obj.episodeRewardBuffer;
            obj.episodeFitness(ep) = finalFitness;
            obj.episodeDuration(ep) = episodeTime;

            % Compute mean losses for episode
            if ~isempty(obj.episodeLossBuffer.actor)
                obj.actorLoss(ep) = mean(obj.episodeLossBuffer.actor);
            end
            if ~isempty(obj.episodeLossBuffer.critic1)
                obj.critic1Loss(ep) = mean(obj.episodeLossBuffer.critic1);
            end
            if ~isempty(obj.episodeLossBuffer.critic2)
                obj.critic2Loss(ep) = mean(obj.episodeLossBuffer.critic2);
            end
            if ~isempty(obj.episodeLossBuffer.alpha)
                obj.alphaValue(ep) = mean(obj.episodeLossBuffer.alpha);
            end
            if ~isempty(obj.episodeLossBuffer.qValues)
                obj.meanQValue(ep) = mean(obj.episodeLossBuffer.qValues);
                obj.maxQValue(ep) = max(obj.episodeLossBuffer.qValues);
            end
            if ~isempty(obj.episodeLossBuffer.entropy)
                obj.policyEntropy(ep) = mean(obj.episodeLossBuffer.entropy);
            end
            if ~isempty(obj.episodeLossBuffer.actorGrad)
                obj.actorGradNorm(ep) = mean(obj.episodeLossBuffer.actorGrad);
            end
            if ~isempty(obj.episodeLossBuffer.criticGrad)
                obj.criticGradNorm(ep) = mean(obj.episodeLossBuffer.criticGrad);
            end

            % Sample efficiency
            if ep == 1
                obj.totalSamples(ep) = obj.episodeIterations;
            else
                obj.totalSamples(ep) = obj.totalSamples(ep-1) + obj.episodeIterations;
            end
            obj.rewardPerSample(ep) = obj.episodeRewards(ep) / obj.episodeIterations;
        end

        function save(obj, filepath)
            % Save training log to MAT file
            logData = struct();
            logData.config = obj.config;
            logData.modelName = obj.modelName;
            logData.startTime = obj.startTime;
            logData.endTime = datetime('now');

            % Episode metrics
            logData.episodeRewards = obj.episodeRewards;
            logData.episodeFitness = obj.episodeFitness;
            logData.episodeDuration = obj.episodeDuration;

            % Loss metrics
            logData.actorLoss = obj.actorLoss;
            logData.critic1Loss = obj.critic1Loss;
            logData.critic2Loss = obj.critic2Loss;
            logData.alphaValue = obj.alphaValue;

            % Q-value and entropy metrics
            logData.meanQValue = obj.meanQValue;
            logData.maxQValue = obj.maxQValue;
            logData.policyEntropy = obj.policyEntropy;

            % Gradient norms
            logData.actorGradNorm = obj.actorGradNorm;
            logData.criticGradNorm = obj.criticGradNorm;

            % Sample efficiency
            logData.totalSamples = obj.totalSamples;
            logData.rewardPerSample = obj.rewardPerSample;

            save(filepath, 'logData');
            fprintf('✓ Training log saved: %s\n', filepath);
        end

        function printSummary(obj)
            % Print training summary statistics
            fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
            fprintf('║              Training Summary: %s\n', pad(obj.modelName, 20));
            fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

            fprintf('Training Duration: %s\n', datetime('now') - obj.startTime);
            fprintf('Episodes Completed: %d / %d\n', obj.currentEpisode, obj.config.numEpisodes);
            fprintf('Total Samples: %d\n', obj.totalSamples(obj.currentEpisode));

            fprintf('\n--- Final Performance ---\n');
            fprintf('Best Fitness: %.2f\n', min(obj.episodeFitness(1:obj.currentEpisode)));
            fprintf('Final Episode Reward: %.2f\n', obj.episodeRewards(obj.currentEpisode));
            fprintf('Mean Episode Duration: %.2f s\n', ...
                mean(obj.episodeDuration(1:obj.currentEpisode)));

            fprintf('\n--- Loss Statistics (Last 10 Episodes) ---\n');
            last10 = max(1, obj.currentEpisode-9):obj.currentEpisode;
            fprintf('Actor Loss: %.4f ± %.4f\n', ...
                mean(obj.actorLoss(last10)), std(obj.actorLoss(last10)));
            fprintf('Critic Loss: %.4f ± %.4f\n', ...
                mean(obj.critic1Loss(last10)), std(obj.critic1Loss(last10)));
            fprintf('Alpha: %.4f ± %.4f\n', ...
                mean(obj.alphaValue(last10)), std(obj.alphaValue(last10)));

            fprintf('\n══════════════════════════════════════════════════════════\n');
        end
    end
end
