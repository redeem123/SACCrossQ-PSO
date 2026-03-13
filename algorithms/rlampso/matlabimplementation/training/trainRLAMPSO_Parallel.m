function [bestAgent, trainingStats] = trainRLAMPSO_Parallel(config, savePathOrAgent)
    % GPU-Optimized Sequential Training for RLAM-PSO
    %
    % Single-GPU sequential training optimized for dlnetwork stability
    %
    % Inputs:
    %   config: Configuration from RLAMPSO_Config()
    %   savePathOrAgent: Either:
    %                    - String: path to save best agent
    %                    - Struct: pretrained agent to continue training
    %                    - Empty: use config.savePath
    %
    % Outputs:
    %   bestAgent: Trained agent
    %   trainingStats: Training statistics
    %
    % Usage:
    %   config = RLAMPSO_Config('baseline');
    %   [agent, stats] = trainRLAMPSO_Parallel(config, 'models/agent.mat');

    % Parse arguments
    if nargin < 2 || isempty(savePathOrAgent)
        savePath = config.savePath;
        pretrainedAgent = [];
    elseif ischar(savePathOrAgent) || isstring(savePathOrAgent)
        savePath = savePathOrAgent;
        pretrainedAgent = [];
    elseif isstruct(savePathOrAgent)
        savePath = config.savePath;
        pretrainedAgent = savePathOrAgent;
    else
        error('Second argument must be a save path (string) or pretrained agent (struct)');
    end

    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════╗\n');
    fprintf('║  RLAM-PSO GPU-Optimized Sequential Training          ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n');
    fprintf('\n');

    % ===== GPU SETUP =====

    if config.useGPU && canUseGPU()
        gpu = gpuDevice;
        fprintf('✓ GPU detected: %s\n', gpu.Name);
        fprintf('  Available Memory: %.1f GB\n', gpu.AvailableMemory / 1e9);
        fprintf('  Batch Size: %d\n\n', config.batchSize);
    elseif config.useGPU
        warning('GPU requested but not available! Falling back to CPU.');
        fprintf('⚠ Training will be slower on CPU\n\n');
        config.useGPU = false;
    else
        fprintf('ℹ Running in CPU mode\n\n');
    end

    % ===== TRAINING CONFIGURATION =====

    fprintf('=== Training Configuration ===\n');
    fprintf('Episodes: %d\n', config.numEpisodes);
    fprintf('Population: %d\n', config.popSize);
    fprintf('Max Iterations: %d\n', config.maxIterations);
    fprintf('Batch Size: %d\n', config.batchSize);
    fprintf('===============================\n\n');

    % ===== INITIALIZE AGENT =====

    if ~isempty(pretrainedAgent)
        fprintf('Loading pretrained agent (Phase 2 fine-tuning)...\n');
        agent = pretrainedAgent;

        % Update learning rates if modified for fine-tuning
        if isfield(agent, 'actorLR')
            agent.actorLR = config.actorLR;
        end
        if isfield(agent, 'criticLR')
            agent.criticLR = config.criticLR;
        end

        fprintf('✓ Pretrained agent loaded\n');

        % ===== MOVE PRETRAINED NETWORKS TO GPU =====
        if config.useGPU && canUseGPU() && isfield(agent, 'useDLToolbox') && agent.useDLToolbox
            fprintf('Moving pretrained networks to GPU...\n');

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

            fprintf('✓ Pretrained networks moved to GPU\n');
        end
        fprintf('\n');
    else
        fprintf('Initializing new agent from scratch...\n');
        agent = initializeAgentAuto(config, '');
        fprintf('✓ Agent initialized\n\n');
    end

    % ===== INITIALIZE CURRICULUM (if enabled) =====

    if config.useCurriculum
        fprintf('Initializing Curriculum Manager...\n');
        curriculumManager = CurriculumManager(config.curriculumSuccessThreshold, ...
                                              config.curriculumWindowSize);
        fprintf('✓ Curriculum Manager initialized\n\n');
    else
        curriculumManager = [];
    end

    % ===== STATISTICS STORAGE =====

    trainingStats = struct();
    trainingStats.episodeRewards = zeros(config.numEpisodes, 1);
    trainingStats.episodeFitness = zeros(config.numEpisodes, 1);
    trainingStats.episodeTimes = zeros(config.numEpisodes, 1);
    trainingStats.episodeIterations = zeros(config.numEpisodes, 1);
    trainingStats.bufferSizes = zeros(config.numEpisodes, 1);

    % ===== SEQUENTIAL TRAINING LOOP =====

    fprintf('Starting GPU-optimized sequential training...\n\n');
    trainingStartTime = tic;

    for episode = 1:config.numEpisodes
        episodeStartTime = tic;

        % Generate scenario
        [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, mapSize] = ...
            generateTrainingScenario(config, curriculumManager, episode);

        % ===== EPISODE-LEVEL REWARDS: Disable mid-episode training =====
        % If using episode-level rewards, we need to collect experiences first
        % then apply the final episode reward retroactively
        if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards
            % Temporarily disable mid-episode training
            originalMidEpisodeTraining = config.enableMidEpisodeTraining;
            config.enableMidEpisodeTraining = false;
        end

        % Run one episode of RLAM-PSO (REUSE agent for speed!)
        [~, convergenceHistory, ~, algorithmStats] = globalPathPlanningRLAMPSO(...
            startPoint, goalPoint, obstacles, trees, ...
            terrainGrid, terrainX, terrainY, mapSize, ...
            config.popSize, config.maxIterations, ...
            0.7, 1.5, 1.5, '', config, agent);

        % Restore mid-episode training setting
        if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards
            config.enableMidEpisodeTraining = originalMidEpisodeTraining;
        end

        % Update agent with trained networks (agent evolves over episodes)
        if isfield(algorithmStats, 'trainedAgent')
            agent = algorithmStats.trainedAgent;
        end

        % ===== EPISODE-LEVEL REWARDS (DISABLED) =====
        % Run baseline comparison and train with episode reward if enabled
        if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards && ...
           isfield(algorithmStats, 'episodeExperiences')

            % ===== BASELINE COMPARISON FOR PROPER CREDIT ASSIGNMENT =====
            % Run baseline PSO on SAME environment to compare performance
            % This solves the credit assignment problem by providing a reference
            %
            % CRITICAL: The network needs to know if its parameter adaptation
            % strategy is better than fixed parameters, not just if it's improving
            [~, ~, baselineStats] = globalPathPlanningPSO(...
                startPoint, goalPoint, obstacles, trees, ...
                terrainGrid, terrainX, terrainY, mapSize, ...
                config.popSize, config.maxIterations, ...
                0.7, 1.5, 1.5);  % Fixed baseline parameters

            % Calculate episode-level reward: improvement over baseline
            rlamFitness = algorithmStats.actualBestFitness;
            baselineFitness = baselineStats.actualBestFitness;
            episodeReward = (baselineFitness - rlamFitness);  % Positive if RLAM better

            % Normalize by baseline to get relative improvement
            if abs(baselineFitness) > 1e-6
                episodeReward = episodeReward / baselineFitness;
            end

            % Debug logging every 10 episodes
            if mod(episode, 10) == 0
                fprintf('[EPISODE REWARD] Ep %d: RLAM=%.2f vs Baseline=%.2f → Reward=%+.4f\n', ...
                        episode, rlamFitness, baselineFitness, episodeReward);
            end

            % Store baseline comparison (for analysis)
            algorithmStats.baselineFitness = baselineFitness;
            algorithmStats.episodeReward = episodeReward;

            % ===== TRAIN WITH EPISODE-LEVEL REWARD =====
            % Add all collected experiences to buffer with episode reward, then train

            episodeExperiences = algorithmStats.episodeExperiences;
            numExperiences = length(episodeExperiences);

            % Add all experiences to replay buffer with episode reward
            for i = 1:numExperiences
                exp = episodeExperiences(i);
                exp.reward = episodeReward;  % All experiences get same episode reward

                % Add to replay buffer
                if agent.usePER
                    agent.replayBuffer.add(exp);
                else
                    if length(agent.replayBuffer) < agent.bufferSize
                        agent.replayBuffer = [agent.replayBuffer, exp];
                    else
                        agent.replayBuffer(agent.bufferIndex) = exp;
                        agent.bufferIndex = mod(agent.bufferIndex, agent.bufferSize) + 1;
                    end
                end
            end

            % Now train network on the buffer (which has experiences with episode reward)
            numTrainingSteps = min(numExperiences, 100);  % Train proportional to experiences

            for trainStep = 1:numTrainingSteps
                if agent.usePER
                    bufferSize = agent.replayBuffer.size();
                else
                    bufferSize = length(agent.replayBuffer);
                end

                if bufferSize >= agent.batchSize
                    % Train network using TD3
                    if agent.useTD3
                        agent = trainDLToolboxTD3(agent);
                    else
                        agent = trainDLToolboxDDPG(agent);
                    end
                end
            end

            if mod(episode, 10) == 0
                fprintf('  → Added %d experiences with episode reward %+.4f, trained %d steps\n', ...
                        numExperiences, episodeReward, numTrainingSteps);
            end
        end

        % Record statistics
        trainingStats.episodeRewards(episode) = algorithmStats.actualBestFitness;
        trainingStats.episodeFitness(episode) = algorithmStats.actualBestFitness;
        trainingStats.episodeTimes(episode) = toc(episodeStartTime);
        trainingStats.episodeIterations(episode) = length(convergenceHistory);

        % Get buffer size
        if isfield(agent, 'usePER') && agent.usePER
            trainingStats.bufferSizes(episode) = agent.replayBuffer.size();
        else
            trainingStats.bufferSizes(episode) = length(agent.replayBuffer);
        end

        % Decay exploration noise AFTER each episode
        if isfield(agent, 'explorationNoise') && isfield(agent, 'noiseDecay') && isfield(agent, 'minNoise')
            agent.explorationNoise = max(agent.minNoise, ...
                                        agent.explorationNoise * agent.noiseDecay);
        end

        % Update curriculum if enabled
        if config.useCurriculum
            % Record episode fitness for curriculum progression
            curriculumManager.recordEpisode(true, trainingStats.episodeFitness(episode));

            % Check if should advance to next difficulty level
            if curriculumManager.checkAdvancement()
                curriculumManager.advance();
            end
        end

        % ===== DETAILED LOGGING EVERY EPISODE (Evidence of Learning) =====
        progress = 100 * episode / config.numEpisodes;
        avgFitness = mean(trainingStats.episodeFitness(max(1, episode-9):episode));
        avgTime = mean(trainingStats.episodeTimes(max(1, episode-9):episode));

        % Calculate reward improvement (negative fitness is better, so higher = better improvement)
        initialFitness = trainingStats.episodeFitness(1);
        fitnessImprovement = initialFitness - algorithmStats.actualBestFitness;
        avgImprovement = initialFitness - avgFitness;

        % Get evidence parameters
        bufferSize = trainingStats.bufferSizes(episode);
        explorationNoise = 0;
        if isfield(agent, 'explorationNoise')
            explorationNoise = agent.explorationNoise;
        end

        % Get recent training losses (if available)
        actorLoss = 0;
        criticLoss = 0;
        if isfield(agent, 'lastActorLoss')
            actorLoss = agent.lastActorLoss;
        end
        if isfield(agent, 'lastCriticLoss')
            criticLoss = agent.lastCriticLoss;
        end

        % Main episode info - showing fitness improvement as reward metric
        fprintf('[Ep %3d/%d] Fit: %7.1f | Avg: %7.1f | Reward: %+6.1f | Time: %5.1fs\n', ...
               episode, config.numEpisodes, ...
               algorithmStats.actualBestFitness, avgFitness, avgImprovement, ...
               trainingStats.episodeTimes(episode));

        % Evidence of learning
        fprintf('          Buffer: %5d | Noise: %.4f', bufferSize, explorationNoise);

        % Show losses if training occurred
        if bufferSize >= agent.batchSize && (actorLoss ~= 0 || criticLoss ~= 0)
            fprintf(' | ActorLoss: %.3e | CriticLoss: %.3e', actorLoss, criticLoss);
        elseif bufferSize < agent.batchSize
            fprintf(' | Status: Warmup (need %d)', agent.batchSize);
        end
        fprintf('\n');

        % GPU check every 5 episodes
        if mod(episode, 5) == 0 && config.useGPU && canUseGPU()
            gpu = gpuDevice;
            memUsed = (gpu.TotalMemory - gpu.AvailableMemory) / 1e9;
            fprintf('          GPU: %.1f / %.1f GB | Utilization: check nvidia-smi\n', ...
                   memUsed, gpu.TotalMemory / 1e9);
        end

        % Periodic saving
        if mod(episode, config.saveInterval) == 0 && ~isempty(savePath)
            checkpointPath = strrep(savePath, '.mat', sprintf('_ep%d.mat', episode));

            % Ensure directory exists - always try to create it
            [checkpointDir, ~, ~] = fileparts(checkpointPath);
            if ~isempty(checkpointDir)
                % Create directory (won't fail if it already exists)
                if ~exist(checkpointDir, 'dir')
                    fprintf('       Creating directory: %s\n', checkpointDir);
                    mkdir(checkpointDir);
                end
            else
                % If no directory in path, use current directory
                checkpointPath = fullfile(pwd, checkpointPath);
                fprintf('       Using full path: %s\n', checkpointPath);
            end

            trainedAgent = agent;
            save(checkpointPath, 'trainedAgent', 'trainingStats', 'config', 'episode');
            fprintf('       ✓ Checkpoint saved: %s\n', checkpointPath);
        end
    end

    totalTrainingTime = toc(trainingStartTime);

    fprintf('\n✓ Sequential training complete!\n');
    fprintf('Total Time: %.2f minutes (%.2f hours)\n', ...
            totalTrainingTime/60, totalTrainingTime/3600);

    % ===== FINAL STATISTICS =====

    fprintf('\n=== Training Summary ===\n');
    fprintf('Total Episodes: %d\n', config.numEpisodes);
    lastWindowImprovement = trainingStats.episodeFitness(1) - mean(trainingStats.episodeFitness(max(1,end-19):end));
    fprintf('Final Reward (Avg Improvement last 20ep): %.2f\n', lastWindowImprovement);
    fprintf('Best Fitness: %.2f\n', min(trainingStats.episodeFitness));
    fprintf('Final Fitness: %.2f\n', trainingStats.episodeFitness(end));
    fprintf('Avg Episode Time: %.2f seconds\n', mean(trainingStats.episodeTimes));
    fprintf('Final Buffer Size: %d experiences\n', trainingStats.bufferSizes(end));
    fprintf('========================\n\n');

    % ===== CURRICULUM STATISTICS =====

    if config.useCurriculum
        fprintf('=== Curriculum Learning Summary ===\n');
        fprintf('Final Level: %d/%d\n', curriculumManager.currentLevel, length(curriculumManager.levels));
        fprintf('Episodes per Level:\n');
        for i = 1:curriculumManager.currentLevel
            fprintf('  Level %d: %d episodes\n', i, curriculumManager.episodesPerLevel(i));
        end
        fprintf('===================================\n\n');
    end

    % ===== SAVE FINAL AGENT =====

    bestAgent = agent;

    if ~isempty(savePath)
        % Ensure directory exists - always try to create it
        [saveDir, ~, ~] = fileparts(savePath);
        if ~isempty(saveDir)
            % Create directory (won't fail if it already exists)
            if ~exist(saveDir, 'dir')
                fprintf('Creating directory: %s\n', saveDir);
                mkdir(saveDir);
            end
        else
            % If no directory in path, use current directory
            savePath = fullfile(pwd, savePath);
            fprintf('Using full path: %s\n', savePath);
        end

        fprintf('Saving final trained agent to: %s\n', savePath);
        trainedAgent = bestAgent;
        save(savePath, 'trainedAgent', 'trainingStats', 'config');
        fprintf('✓ Final agent saved\n');
    end

    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════╗\n');
    fprintf('║       GPU-Optimized Training Complete!                ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end

%% Helper Functions

function net = moveNetworkToGPU_local(net)
    % Properly move dlnetwork to GPU by converting all learnables
    learnables = net.Learnables;
    for i = 1:height(learnables)
        learnables.Value{i} = gpuArray(learnables.Value{i});
    end
    net.Learnables = learnables;
end

function [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, mapSize] = ...
    generateTrainingScenario(config, curriculumManager, episode)
    % Generate training scenario with optional curriculum

    mapSize = config.mapSize;

    % Generate terrain
    [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, 1);

    % Generate start and goal (scaled from benchmark)
    scaleFactor = mapSize(1) / 100;
    startPoint = [10 * scaleFactor, 95 * scaleFactor, 10];
    goalPoint = [97 * scaleFactor, 2 * scaleFactor, 10];

    % Determine difficulty
    if ~isempty(curriculumManager)
        % Use curriculum level - sample from range
        levelDef = curriculumManager.levels{curriculumManager.currentLevel};

        % Sample number of obstacles and trees from level's range
        numObstacles = randi([levelDef.obstacleRange(1), levelDef.obstacleRange(2)]);
        numTrees = randi([levelDef.treeRange(1), levelDef.treeRange(2)]);
    else
        % Progressive difficulty based on episode
        episodeFraction = min(episode / config.numEpisodes, 1.0);
        numTrees = round(10 + episodeFraction * 80);      % 10 → 90
        numObstacles = round(5 + episodeFraction * 10);   % 5 → 15
    end

    % Generate environment
    trees = generateFixedTrees(numTrees, terrainGrid, terrainX, terrainY, mapSize);
    obstacles = generateFixedObstacles(numObstacles, mapSize, terrainGrid, terrainX, terrainY);
end
