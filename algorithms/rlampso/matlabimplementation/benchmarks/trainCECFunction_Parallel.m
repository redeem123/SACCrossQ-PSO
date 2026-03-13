function [trainedAgent, stats] = trainCECFunction_Parallel(functionID, numEpisodes, config, preloadedAgent)
    % GPU-OPTIMIZED SEQUENTIAL TRAINING ON SINGLE CEC BENCHMARK FUNCTION
    %
    % Trains RLAMPSO agent on one CEC function using single GPU worker
    % Faster and more stable than parallel training for dlnetwork
    %
    % Inputs:
    %   functionID: CEC function number (1-28)
    %   numEpisodes: Total episodes to run
    %   config: RLAMPSO configuration
    %   preloadedAgent: Pre-initialized agent to continue training
    %
    % Outputs:
    %   trainedAgent: Agent after training on this function
    %   stats: Training statistics

    if nargin < 4, preloadedAgent = []; end

    % Initialize agent if not provided
    if isempty(preloadedAgent)
        agent = initializeAgentAuto(config, '');
    else
        agent = preloadedAgent;

        % ===== MOVE PRELOADED AGENT NETWORKS TO GPU =====
        if config.useGPU && canUseGPU() && isfield(agent, 'useDLToolbox') && agent.useDLToolbox
            % Move all networks to GPU - proper method
            if isfield(agent, 'actor')
                agent.actor = moveNetworkToGPU_local(agent.actor);
            end
            if isfield(agent, 'critic')
                agent.critic = moveNetworkToGPU_local(agent.critic);
            end
            if isfield(agent, 'targetActor')
                agent.targetActor = moveNetworkToGPU_local(agent.targetActor);
            end
            if isfield(agent, 'targetCritic')
                agent.targetCritic = moveNetworkToGPU_local(agent.targetCritic);
            end

            % TD3 has twin critics
            if isfield(agent, 'critic1')
                agent.critic1 = moveNetworkToGPU_local(agent.critic1);
            end
            if isfield(agent, 'critic2')
                agent.critic2 = moveNetworkToGPU_local(agent.critic2);
            end
            if isfield(agent, 'targetCritic1')
                agent.targetCritic1 = moveNetworkToGPU_local(agent.targetCritic1);
            end
            if isfield(agent, 'targetCritic2')
                agent.targetCritic2 = moveNetworkToGPU_local(agent.targetCritic2);
            end
        end
    end

    % CEC benchmark parameters
    dimensions = 30;  % Standard for CEC2013
    [~, optimalValue] = CECBenchmarks(zeros(dimensions, 1), functionID);

    % ===== TRANSFER LEARNING FIX: Reset per-function episode counter =====
    % This ensures Q-value clamping starts fresh for each new function
    % Prevents explosion when moving from F1 to F2, F2 to F3, etc.
    if isfield(agent, 'functionEpisodeCount')
        fprintf('  Resetting functionEpisodeCount (was %d)\n', agent.functionEpisodeCount);
        agent.functionEpisodeCount = 0;
    end

    % Initialize statistics arrays
    stats = struct();
    stats.episodeFitness = zeros(numEpisodes, 1);
    stats.episodeTimes = zeros(numEpisodes, 1);
    stats.episodeSuccess = false(numEpisodes, 1);

    % Sequential training loop
    for episode = 1:numEpisodes
        episodeStartTime = tic;

        % Run RLAMPSO on CEC function
        [bestFitness, ~, agent] = runRLAMPSOonCEC(...
            functionID, dimensions, config, agent);

        episodeTime = toc(episodeStartTime);

        % Store statistics
        stats.episodeFitness(episode) = bestFitness;
        stats.episodeTimes(episode) = episodeTime;
        stats.episodeSuccess(episode) = abs(bestFitness - optimalValue) < 1e-6;

        % Decay exploration noise AFTER each episode
        if isfield(agent, 'explorationNoise') && isfield(agent, 'noiseDecay') && isfield(agent, 'minNoise')
            agent.explorationNoise = max(agent.minNoise, ...
                                        agent.explorationNoise * agent.noiseDecay);
        end

        % ===== TRANSFER LEARNING FIX: Increment per-function episode counter =====
        if isfield(agent, 'functionEpisodeCount')
            agent.functionEpisodeCount = agent.functionEpisodeCount + 1;
        end

        % ===== DETAILED LOGGING EVERY EPISODE (Evidence of Learning) =====
        progress = 100 * episode / numEpisodes;
        avgFitness = mean(stats.episodeFitness(max(1, episode-9):episode));

        % Get buffer size
        if isfield(agent, 'usePER') && agent.usePER
            bufferSize = agent.replayBuffer.size();
        else
            bufferSize = length(agent.replayBuffer);
        end

        % Get exploration noise
        explorationNoise = 0;
        if isfield(agent, 'explorationNoise')
            explorationNoise = agent.explorationNoise;
        end

        % Get losses
        actorLoss = 0;
        criticLoss = 0;
        if isfield(agent, 'lastActorLoss')
            actorLoss = agent.lastActorLoss;
        end
        if isfield(agent, 'lastCriticLoss')
            criticLoss = agent.lastCriticLoss;
        end

        % Main episode info
        fprintf('  [Ep %3d/%d] F%d | Fitness: %.4e | Avg: %.4e | Time: %5.1fs\n', ...
               episode, numEpisodes, functionID, bestFitness, avgFitness, episodeTime);

        % Evidence of learning
        fprintf('            Buffer: %5d | Noise: %.4f', bufferSize, explorationNoise);

        % Show losses if training occurred
        if bufferSize >= agent.batchSize && (actorLoss ~= 0 || criticLoss ~= 0)
            fprintf(' | ActorLoss: %.3e | CriticLoss: %.3e', actorLoss, criticLoss);
        elseif bufferSize < agent.batchSize
            fprintf(' | Status: Warmup (need %d)', agent.batchSize);
        end
        fprintf('\n');
    end

    % Calculate summary statistics
    stats.finalFitness = stats.episodeFitness(end);
    stats.bestFitness = min(stats.episodeFitness);
    stats.avgFitness = mean(stats.episodeFitness);
    stats.avgTime = mean(stats.episodeTimes);
    stats.successRate = sum(stats.episodeSuccess) / numEpisodes;

    trainedAgent = agent;

    fprintf('\n✓ F%d training complete: Best=%.4e | Avg=%.4e | Success=%.1f%%\n', ...
            functionID, stats.bestFitness, stats.avgFitness, stats.successRate*100);
end

function net = moveNetworkToGPU_local(net)
    % Properly move dlnetwork to GPU by converting all learnables
    learnables = net.Learnables;
    for i = 1:height(learnables)
        learnables.Value{i} = gpuArray(learnables.Value{i});
    end
    net.Learnables = learnables;
end
